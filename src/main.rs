use std::{
    cmp::max,
    sync::{Arc, Mutex},
    time::Duration,
};
use tracing::{debug, error, info};

use anyhow::Result;
use cached_path::Cache;
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, SampleRate, SupportedBufferSize, SupportedStreamConfig,
};
use crossbeam_channel::bounded;
use enigo::{Enigo, Key, KeyboardControllable};
use itertools::Itertools;
use url::Url;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The model name to download from Huggingface (see https://huggingface.co/ggerganov/whisper.cpp/tree/main)
    #[arg(short, long, default_value = "ggml-tiny.en.bin")]
    model: String,

    /// How often to inference and type. Set to 0 to continuously inference.
    #[arg(short, long, default_value_t = 0.0)]
    period: f32,

    /// The size of the Whisper beam search
    #[arg(short, long, default_value_t = 3)]
    beam_search_size: u8,
}

fn main() -> Result<(), anyhow::Error> {
    let args = Args::parse();

    let url = Url::parse("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/")
        .unwrap()
        .join(&args.model)
        .unwrap();
    info!("Fetching model from {}", url.as_str());
    let model_path = Cache::builder()
        .dir(dirs::cache_dir().unwrap().join("lausch/"))
        .freshness_lifetime(u64::MAX)
        .build()
        .unwrap()
        .cached_path(url.as_str())
        .expect("Failed to fetch model from repository");
    debug!("Loading model from {model_path:?}");

    let audio_data = Arc::new(Mutex::new(vec![0_f32]));
    let (exit_tx, exit_rx) = bounded::<()>(1);

    let audio_data_clone = audio_data.clone();
    let record_thread = std::thread::spawn(move || {
        let audio_data = audio_data_clone;

        debug!(
            "Audio devices: {}",
            cpal::available_hosts()
                .iter()
                .map(|host| host.name())
                .collect::<Vec<&str>>()
                .join(", ")
        );
        let device: cpal::Device = cpal::default_host()
            .default_input_device()
            .expect("Failed to retrieve default input audio device");
        debug!("Chose device: {}", device.name().unwrap());
        let stream = device
            .build_input_stream(
                &SupportedStreamConfig::new(
                    1,
                    SampleRate(16000),
                    SupportedBufferSize::Unknown,
                    SampleFormat::F32,
                )
                .into(),
                move |chunk: &[f32], _: &_| {
                    let mut audio_data_mut = audio_data.lock().unwrap();
                    let audio_data: &mut Vec<f32> = audio_data_mut.as_mut();
                    audio_data.extend_from_slice(chunk);
                },
                move |err| error!("An error occurred while recording: {}", err),
                None,
            )
            .expect("Failed to build input stream");
        debug!("Recording from audio device");
        stream.play().expect("Failed to start input stream");
        exit_rx.recv().expect("Failed to receive exit signal");
    });

    let audio_data_clone = audio_data.clone();
    let inference_thread = std::thread::spawn(move || {
        let audio_data = audio_data_clone;
        let mut whisper =
            WhisperContext::new(model_path.to_str().unwrap()).expect("Failed to load model");

        let mut enigo = Enigo::new();
        let mut all_text: String = "".into();
        loop {
            std::thread::sleep(Duration::from_secs_f32(args.period));
            let audio_data_mut = audio_data.lock().expect("Failed to lock audio_data");
            let audio_data = audio_data_mut.clone();
            drop(audio_data_mut);

            let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                beam_size: 3,
                patience: 0.,
            });
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            whisper
                .full(params, &audio_data[..])
                .expect("Failed to inference");
            let mut tokens = vec![];
            for segment in 0..whisper.full_n_segments() {
                for token in 0..whisper.full_n_tokens(segment) {
                    let token = whisper.full_get_token_data(segment, token);
                    let token_str = whisper
                        .token_to_str(token.id)
                        .expect("Could not convert token to string");
                    if token_str.starts_with("[") && token_str.ends_with("]") {
                        continue;
                    }
                    if token.id != whisper.token_eot() {
                        tokens.push(token)
                    }
                }
            }
            let tokens = (0..whisper.full_n_segments())
                .flat_map(|segment_idx| {
                    (0..whisper.full_n_tokens(segment_idx))
                        .map(|token_idx| whisper.full_get_token_data(segment_idx, token_idx))
                        .collect_vec()
                })
                .collect_vec();
            let tokens_str = tokens
                .iter()
                .map(|token| {
                    whisper
                        .token_to_str(token.id)
                        .expect("Could not decode token")
                })
                .filter(|token_str| !(token_str.starts_with("[") || token_str.ends_with("]")))
                .collect_vec();
            let text = tokens_str.join("");
            let text = text
                .replace(" [BLANK_AUDIO]", "")
                .replace(" [BLANK_AUDIO", "");
            let text = text.trim();

            let longest_common_prefix = all_text
                .char_indices()
                .find(|&(i, c)| text[i..].chars().next() != Some(c))
                .map_or(all_text.clone(), |(i, _)| all_text[..i].into());
            let num_to_delete = all_text.len() as i32 - longest_common_prefix.len() as i32;
            let mut last_text = all_text.clone();
            if num_to_delete > 0 {
                for _ in 0..num_to_delete {
                    enigo.key_click(Key::Backspace);
                }
                last_text =
                    all_text[0..max(all_text.len() as i32 - num_to_delete, 0) as usize].into();
            }
            let new_text: String = text[longest_common_prefix.len()..text.len()].into();

            let should_exit = tokens
                .last()
                .map_or(false, |token| token.id == whisper.token_eot());

            debug!("Typing {new_text:?}");
            enigo.key_sequence(new_text.as_str());
            all_text = format!("{last_text}{new_text}");

            if should_exit {
                exit_tx.send(()).expect("Failed to sent exit signal");
                return;
            }
        }
    });

    record_thread.join().unwrap();
    inference_thread.join().unwrap();

    Ok(())
}

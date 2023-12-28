use anyhow::{Error, Result};
use byteorder::LittleEndian;
use byteorder::WriteBytesExt;
use cached_path::Cache;
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, SampleRate, SupportedBufferSize, SupportedStreamConfig,
};
use crossbeam_channel::bounded;
use inputbot::get_keybd_key;

use inputbot::KeybdKey;
use itertools::Itertools;
use lazy_regex::regex;
use ndarray::{s, Array1, Array3, ArrayBase, Axis, CowArray, Dim, Ix1, Ix3, OwnedRepr};
use ndarray_stats::{interpolate::Midpoint, Quantile1dExt};
use noisy_float::types::{n32, n64, N32};
use ort::tensor::OrtOwnedTensor;
use std::thread::sleep;
use std::{
    cmp::max,
    fs::File,
    sync::{Arc, Mutex, RwLock},
    time::{self, Duration},
};
use tracing::Level;
use tracing::{debug, error, info};
use url::Url;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperTokenData,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Logging verbosity
    #[arg(short, long, default_value = "info")]
    verbosity: Level,

    /// The model name to download and cache from Huggingface. (see https://huggingface.co/ggerganov/whisper.cpp/tree/main)
    #[arg(short, long, default_value = "ggml-tiny.en.bin")]
    model: String,

    /// The delay between voice activity detections. Set to 0 to continuously detect. The lower the value the higher the CPU usage.
    #[arg(long, default_value_t = 0.1)]
    vad_delay: f32,

    /// The delay between transcription events. Set to 0 to continuously transcribe. The lower the value the higher the CPU usage.
    #[arg(long, default_value_t = 0.25)]
    transcribe_delay: f32,

    /// Sensitivity for the Silero VAD.
    #[arg(long, default_value_t = 0.5)]
    vad_sensitivity: f32,

    /// The size of the Whisper beam search
    #[arg(long, default_value_t = 3)]
    beam_search_size: u8,
}

struct SileroVadSession {
    session: ort::Session,
    sr: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    h: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
    c: ArrayBase<OwnedRepr<f32>, Dim<[usize; 3]>>,
}

impl SileroVadSession {
    fn new(sample_rate: i64) -> Result<SileroVadSession> {
        let model_path = Cache::builder()
            .dir(dirs::cache_dir().unwrap().join("lausch/"))
            .freshness_lifetime(u64::MAX)
            .build()
            .unwrap()
            .cached_path("https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx")
            .expect("Failed to fetch VAD model from repository");
        let session = ort::session::SessionBuilder::new(
            &ort::Environment::builder()
                .with_name("test")
                .with_log_level(ort::LoggingLevel::Verbose)
                .build()?
                .into_arc(),
        )?
        .with_model_from_file(model_path)?;
        Ok(SileroVadSession {
            session,
            sr: Array1::<i64>::from(vec![sample_rate]),
            h: Array3::<f32>::zeros((2, 1, 64)),
            c: Array3::<f32>::zeros((2, 1, 64)),
        })
    }

    fn call(&mut self, input: Array1<f32>) -> Result<f32> {
        let input = input.insert_axis(Axis(0));
        let input = CowArray::from(input).into_dyn();
        let sr = CowArray::from(self.sr.clone()).into_dyn();
        let h = CowArray::from(self.h.clone()).into_dyn();
        let c = CowArray::from(self.c.clone()).into_dyn();
        let inputs = vec![
            ort::Value::from_array(self.session.allocator(), &input)?,
            ort::Value::from_array(self.session.allocator(), &sr)?,
            ort::Value::from_array(self.session.allocator(), &h)?,
            ort::Value::from_array(self.session.allocator(), &c)?,
        ];
        let outputs: Vec<ort::Value> = self.session.run(inputs)?;
        let output: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
        let output = output.view().to_owned()[[0, 0]];
        let h_out: OrtOwnedTensor<f32, _> = outputs[1].try_extract()?;
        let h_out = h_out.view().to_owned().into_dimensionality::<Ix3>()?;
        let c_out: OrtOwnedTensor<f32, _> = outputs[2].try_extract()?;
        let c_out = c_out.view().to_owned().into_dimensionality::<Ix3>()?;

        self.h = h_out;
        self.c = c_out;

        Ok(output)
    }
}

fn main() -> Result<(), Error> {
    let args = Args::parse();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(args.verbosity)
            .finish(),
    )?;
    inputbot::init_device();

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
        .expect("Failed to fetch Whisper model from repository");
    debug!("Loading model from {model_path:?}");

    let audio_data_arc = Arc::new(RwLock::new(vec![0_f32]));
    let (stop_recording_tx, stop_recording_rx) = bounded::<()>(1);
    let stop_transcription_arc = Arc::new(Mutex::new(false));
    let voice_started_arc = Arc::new(RwLock::new(false));

    let audio_data = audio_data_arc.clone();
    let stop_transcription = stop_transcription_arc.clone();
    let voice_started = voice_started_arc.clone();
    let record_thread = std::thread::spawn(move || {
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
                    let mut audio_data_mut =
                        audio_data.write().expect("Could not write to audio_data");
                    let audio_data: &mut Vec<f32> = audio_data_mut.as_mut();
                    audio_data.extend_from_slice(chunk);
                },
                move |err| error!("An error occurred while recording: {}", err),
                None,
            )
            .expect("Failed to build input stream");
        debug!("Recording from audio device");
        stream.play().expect("Failed to start input stream");
        stop_recording_rx
            .recv()
            .expect("Failed to receive exit signal");
        debug!("Record thread exiting");
        *stop_transcription.lock().unwrap() = true;
    });

    let audio_data = audio_data_arc.clone();
    let vad_thread = std::thread::spawn(move || -> Result<()> {
        let mut vad = SileroVadSession::new(16000)?;
        let vad_scores = &mut Vec::<N32>::new();
        loop {
            let audio_data = {
                let audio_data_ref = audio_data.read().expect("Failed to lock audio_data");
                audio_data_ref.clone()
            };

            // Using the recommended chunk size from: https://github.com/snakers4/silero-vad/blob/5e7ee10ee065ab2b98751dd82b28e3c6360e19aa/utils_vad.py#L207C60-L207C64
            let vad_input = if audio_data.len() < 1536 {
                let to_pad = Array1::from(audio_data);
                let mut padded = Array1::<f32>::zeros(Ix1(1536));
                padded
                    .slice_mut(s![-(to_pad.len() as i32)..])
                    .assign(&to_pad);
                padded
            } else {
                Array1::from(audio_data).slice(s![-1536..]).to_owned()
            };

            let t0 = time::Instant::now();
            let vad_score = n32(vad.call(vad_input).expect("VAD failed"));
            debug!(
                "VAD score {vad_score}, inference duration {:?}",
                time::Instant::now().duration_since(t0)
            );
            vad_scores.push(vad_score);
            let mut vad_scores_arr = Array1::from_vec(vad_scores.to_vec());
            let vad_median = if vad_scores_arr.len() < 5 {
                None
            } else {
                Some(
                    vad_scores_arr
                        .slice_mut(s![-5..])
                        .quantile_mut(n64(0.5), &Midpoint)?,
                )
            };
            if voice_started.read().unwrap().clone() {
                if vad_median.is_some_and(|i| i < args.vad_sensitivity) {
                    debug!("VAD thread exiting");
                    stop_recording_tx
                        .send(())
                        .expect("Failed to send recording exit signal");
                    return Ok(());
                }
            } else {
                if vad_median.is_some_and(|i| i >= args.vad_sensitivity) {
                    debug!("VAD detected start");
                    *voice_started.write().unwrap() = true;
                }
            };
            std::thread::sleep(Duration::from_secs_f32(args.vad_delay));
        }
    });

    let audio_data = audio_data_arc.clone();
    let stop_transcription = stop_transcription_arc.clone();
    let voice_started = voice_started_arc.clone();
    let inference_thread = std::thread::spawn(move || -> Result<()> {
        let whisper_model = WhisperContext::new_with_params(
            model_path.to_str().unwrap(),
            WhisperContextParameters { use_gpu: true },
        )
        .expect("Failed to load model");
        let mut whisper = whisper_model
            .create_state()
            .expect("Failed to create state");

        let mut all_text: String = "".into();
        let mut exit_next_loop = false;
        let mut exit = false;
        while !exit {
            if !voice_started.read().unwrap().clone() {
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
            let audio_data_mut = audio_data.read().expect("Failed to lock audio_data");
            let audio_data = audio_data_mut.clone();
            drop(audio_data_mut);

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 0 });
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);
            params.set_suppress_blank(true);
            params.set_suppress_non_speech_tokens(false);

            let t0 = time::Instant::now();
            whisper
                .full(params, &audio_data[..])
                .expect("Failed to inference");
            debug!(
                "Whisper inference duration {:?}",
                time::Instant::now().duration_since(t0)
            );
            let mut tokens = vec![];
            for segment in 0..whisper.full_n_segments()? {
                for token in 0..whisper.full_n_tokens(segment)? {
                    let token = whisper.full_get_token_data(segment, token)?;
                    let token_str = match whisper_model.token_to_str(token.id) {
                        Ok(token_str) => token_str,
                        Err(_) => {
                            continue;
                        }
                    };
                    if token_str.starts_with("[") && token_str.ends_with("]") {
                        continue;
                    }
                    if token.id != whisper_model.token_eot() {
                        tokens.push(token)
                    }
                }
            }
            let tokens: Vec<WhisperTokenData> = (0..whisper.full_n_segments()?)
                .flat_map(|segment_idx| {
                    (0..whisper.full_n_tokens(segment_idx).unwrap())
                        .map(|token_idx| {
                            whisper.full_get_token_data(segment_idx, token_idx).unwrap()
                        })
                        .collect_vec()
                })
                .collect_vec();
            let tokens_str = tokens
                .iter()
                .filter_map(|token| whisper_model.token_to_str(token.id).ok())
                .collect_vec();
            let mut text = tokens_str
                .iter()
                .filter(|token_str| !(token_str.starts_with("[") || token_str.ends_with("]")))
                .join("");
            text = regex!(r#" ?\(.+?\)"#).replace_all(&text, "").to_string();
            text = regex!(r#" ?\[.+?\]"#).replace_all(&text, "").to_string();
            text = regex!(r#"\(.+( |$)"#).replace_all(&text, "").to_string();
            text = regex!(r#"\[.+( |$)"#).replace_all(&text, "").to_string();
            let text = text.trim();

            let longest_common_prefix = all_text
                .char_indices()
                .find(|&(i, c)| text[i..].chars().next() != Some(c))
                .map_or(all_text.clone(), |(i, _)| all_text[..i].into());
            let num_to_delete = all_text.len() as i32 - longest_common_prefix.len() as i32;
            let mut last_text = all_text.clone();
            if num_to_delete > 0 {
                debug!("Deleting {num_to_delete}");
                for _ in 0..num_to_delete {
                    press_backspace();
                }
                last_text =
                    all_text[0..max(all_text.len() as i32 - num_to_delete, 0) as usize].into();
            }
            let new_text = text[longest_common_prefix.len()..text.len()].to_string();

            if !new_text.is_empty() {
                debug!("Typing {new_text:?}");
                type_text(&new_text);
                all_text = format!("{last_text}{new_text}");
            }

            if exit_next_loop {
                exit = true;
            }
            if *stop_transcription.lock().unwrap() {
                exit_next_loop = true;
            }
            std::thread::sleep(Duration::from_secs_f32(args.transcribe_delay));
        }
        debug!("Transcription thread exiting");

        let mut f = File::create("/tmp/output.file")?;
        for float in audio_data.read().unwrap().iter() {
            f.write_f32::<LittleEndian>(*float)?;
        }
        Ok(())
    });

    record_thread.join().unwrap();
    inference_thread.join().unwrap()?;
    vad_thread.join().unwrap()?;

    Ok(())
}

fn press_backspace() {
    KeybdKey::BackspaceKey.press();
    KeybdKey::BackspaceKey.release();
}

fn type_text(text: &str) {
    for c in text.chars() {
        let mut uppercase = false;

        if let Some(keybd_key) = {
            if c.is_uppercase()
                || [
                    '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+', '{', '}', '|', ':',
                    '"', '<', '>', '?', '~',
                ]
                .contains(&c)
            {
                uppercase = true;
            }

            get_keybd_key(c)
        } {
            if uppercase {
                KeybdKey::LShiftKey.press();
            }

            keybd_key.press();
            sleep(Duration::from_millis(20));
            keybd_key.release();

            if uppercase {
                KeybdKey::LShiftKey.release();
            }
        };
    }
}

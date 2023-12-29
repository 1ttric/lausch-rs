use anyhow::{Error, Result};
use byteorder::LittleEndian;
use byteorder::WriteBytesExt;
use cached_path::Cache;
use clap::Parser;
use clap::ValueEnum;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, SampleRate, SupportedBufferSize, SupportedStreamConfig,
};
use crossbeam_channel::bounded;
use inputbot::KeySequence;
use inputbot::KeybdKey;
use itertools::Itertools;
use lazy_regex::regex;
use ndarray::{s, Array1, Array3, ArrayBase, Axis, Dim, Ix1, Ix3, OwnedRepr};
use ndarray_stats::{interpolate::Midpoint, Quantile1dExt};
use noisy_float::types::{n32, n64, N32};
use ort::inputs;
use ort::Session;
use std::cmp::min;
use std::io;
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

static AUDIO_SAMPLE_RATE: u32 = 16000;
static VAD_CHUNK_SIZE: u32 = 1536;

#[derive(ValueEnum, Debug, Clone, PartialEq)]
enum Mode {
    Stdout,
    StdoutContinuous,
    Type,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Program mode
    #[arg(value_enum, short, long, default_value_t = Mode::Stdout)]
    mode: Mode,

    /// Logging verbosity
    #[arg(short, long, default_value = "info")]
    verbosity: Level,

    /// The ASR model name to download and cache from Huggingface. (see https://huggingface.co/ggerganov/whisper.cpp/tree/main)
    #[arg(long, default_value = "ggml-tiny.en.bin")]
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

    /// Will save the raw f32le PCM audio as recorded
    #[arg(long)]
    save_audio: Option<String>,
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

        ort::init().commit()?;
        let session = Session::builder()?.with_model_from_file(model_path)?;
        Ok(SileroVadSession {
            session,
            sr: Array1::<i64>::from(vec![sample_rate]),
            h: Array3::<f32>::zeros((2, 1, 64)),
            c: Array3::<f32>::zeros((2, 1, 64)),
        })
    }

    fn call(&mut self, input: Array1<f32>) -> Result<f32> {
        let input = input.insert_axis(Axis(0));
        let outputs = self
            .session
            .run(inputs![
                input,
                self.sr.view(),
                self.h.view(),
                self.c.view()
            ]?)
            .unwrap();
        let output = outputs[0].extract_tensor::<f32>()?;
        let output = output.view().to_owned()[[0, 0]];
        let h_out = outputs[1].extract_tensor::<f32>()?;
        let h_out = h_out.view().to_owned().into_dimensionality::<Ix3>()?;
        let c_out = outputs[2].extract_tensor::<f32>()?;
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
            .with_writer(io::stderr)
            .with_max_level(args.verbosity)
            .finish(),
    )?;

    let url = Url::parse("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/")
        .unwrap()
        .join(&args.model)
        .unwrap();
    debug!("Fetching model from {url}");
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
                    SampleRate(AUDIO_SAMPLE_RATE),
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
        // Checks for a numeric VAD score [0-1] on the current audio buffer at each loop. Once the median of the last 5 scores passes a configurable threshold, consider voice started.
        let mut vad = SileroVadSession::new(AUDIO_SAMPLE_RATE.into())?;
        let vad_scores = &mut Vec::<N32>::new();
        loop {
            let audio_data_vec = {
                let audio_data_ref = audio_data.read().expect("Failed to lock audio_data");
                audio_data_ref.clone()
            };

            // Using the recommended chunk size from: https://github.com/snakers4/silero-vad/blob/5e7ee10ee065ab2b98751dd82b28e3c6360e19aa/utils_vad.py#L207C60-L207C64
            let vad_input = if audio_data_vec.len() < VAD_CHUNK_SIZE as usize {
                let to_pad = Array1::from(audio_data_vec.clone());
                let mut padded = Array1::<f32>::zeros(Ix1(VAD_CHUNK_SIZE as usize));
                padded
                    .slice_mut(s![-(to_pad.len() as i32)..])
                    .assign(&to_pad);
                padded
            } else {
                Array1::from(audio_data_vec.clone())
                    .slice(s![-(VAD_CHUNK_SIZE as i32)..])
                    .to_owned()
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
                    let audio_data_vec_len = audio_data_vec.len();
                    // Once voice is detected, trim the audio buffer to avoid Whisper picking up things the VAD may have missed
                    *audio_data.write().unwrap() = Array1::from(audio_data_vec)
                        .slice(s![-min(
                            VAD_CHUNK_SIZE as i32 * 5,
                            audio_data_vec_len as i32
                        )..])
                        .to_vec();
                }
            };
            std::thread::sleep(Duration::from_secs_f32(args.vad_delay));
        }
    });

    let audio_data = audio_data_arc.clone();
    let stop_transcription = stop_transcription_arc.clone();
    let voice_started = voice_started_arc.clone();
    let inference_thread = std::thread::spawn(move || -> Result<()> {
        // Once voice activity has started, begin transcribing it continuously with a configurable delay. Remove undesired tokens like [UNKNOWN], (music), (unintelligible), and then type the text.
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
                    let _token_str = match whisper_model.token_to_str(token.id) {
                        Ok(token_str) => token_str,
                        Err(_) => {
                            continue;
                        }
                    };
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
                if args.mode == Mode::Type {
                    debug!("Deleting {num_to_delete} characters");
                    for _ in 0..num_to_delete {
                        press_backspace();
                    }
                }
                last_text =
                    all_text[0..max(all_text.len() as i32 - num_to_delete, 0) as usize].into();
            }
            let new_text = text[longest_common_prefix.len()..text.len()].to_string();

            if !new_text.is_empty() {
                if args.mode == Mode::Type {
                    debug!("Typing characters {new_text:?}");
                    type_text(&new_text);
                }
                all_text = format!("{last_text}{new_text}");
            }

            if exit_next_loop {
                exit = true;
            }
            if *stop_transcription.lock().unwrap() {
                exit_next_loop = true;
            }
            debug!("Current text: {all_text:?}");
            if args.mode == Mode::StdoutContinuous {
                println!("{all_text}");
            }
            std::thread::sleep(Duration::from_secs_f32(args.transcribe_delay));
        }
        if args.mode == Mode::Stdout {
            println!("{all_text}");
        }
        debug!("Transcription thread exiting");

        if let Some(path) = args.save_audio {
            let mut f = File::create(path)?;
            for float in audio_data.read().unwrap().iter() {
                f.write_f32::<LittleEndian>(*float)?;
            }
            info!("Audio saved to lausch-debug.pcm");
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
    KeySequence(text).send();
}

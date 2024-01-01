# Lausch

Lausch is a Rust tool that allows a user to dictate text via voice for input and either output the transcribed text or type the text back via simulated keyboard events

[Silero VAD](https://github.com/snakers4/silero-vad) is used to wait for a voice command to begin, and transcription is performed with [Whisper](https://github.com/tazz4843/whisper-rs)

## Installation

```bash
cargo install --git https://git.svc.vesey.tech/will/lausch
```

## Usage

If key inputs are not being sent, you may need to refer to [inputbot](https://github.com/obv-mikhail/InputBot) documentation for additional permissions configuration.

```bash
lausch
```


# VOX — Voice Operated eXecutive

> Local-first AI voice assistant powered by your GPU. Your hardware, your data.

VOX runs LLM inference, speech recognition, and voice synthesis directly on your GPU via Ollama, Faster-Whisper, and Piper/XTTS. No cloud AI APIs. No per-token billing. No data leaves your machine during inference.

**How it works:** Ollama serves a quantized LLM from VRAM on your RTX 3090 (~112 tokens/sec). Whisper transcribes your voice on-GPU. Piper or XTTS synthesizes speech on-GPU. The only network calls are optional tools you choose to enable (weather, etc).

## Features

- **Wake Word Detection** — Always-listening trigger phrase (Porcupine, runs on CPU)
- **Speech-to-Text** — GPU-accelerated transcription (Faster-Whisper on CUDA)
- **LLM Reasoning** — Local model on your GPU via Ollama, with tool calling
- **Text-to-Speech** — Natural voice synthesis (Piper for speed / XTTS v2 for quality)
- **Tool Calling** — Extensible function registry for real-world actions
- **Fast** — Sub-second LLM responses, model stays hot in VRAM

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3090)
- [Ollama](https://ollama.com/) installed and running
- Microphone and speakers

## Quick Start

```bash
# Clone
git clone https://github.com/rayketcham-lab/vox.git
cd vox

# Set up environment
cp .env.example .env
# Edit .env with your settings

# Install
pip install -e ".[dev]"

# Pull an LLM model
ollama pull llama3.2

# Run in text mode (no mic needed)
vox --text

# Run with voice
vox --no-wake    # skip wake word, press Enter to speak
vox              # full wake word mode
```

## Architecture

```
Microphone → Wake Word (CPU) → STT (GPU) → LLM (GPU) → TTS (GPU) → Speaker
                Porcupine      Whisper      Ollama      Piper/XTTS
```

All components run locally. VRAM budget (~24GB):
- Whisper: ~1-2 GB
- LLM (7-13B): ~6-15 GB
- TTS: ~1-2 GB

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for available options.

## Project Status

**v0.1.0** — Project scaffold. Core voice loop in development.

See [Issues](https://github.com/rayketcham-lab/vox/issues) for the development roadmap.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [SECURITY.md](SECURITY.md) for our security policy.

## License

MIT — see [LICENSE](LICENSE).

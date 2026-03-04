# VOX — Voice Operated eXecutive

> Fully local AI voice assistant powered by your GPU. Zero cloud. Total privacy.

VOX is a local-first voice assistant that runs entirely on your hardware. No cloud APIs, no telemetry, no data leaves your network. Wake it with your voice, ask questions, control your environment — all processed locally on your GPU.

## Features

- **Wake Word Detection** — Always-listening trigger phrase (Porcupine)
- **Speech-to-Text** — GPU-accelerated transcription (Faster-Whisper)
- **LLM Reasoning** — Local language model with tool calling (Ollama)
- **Text-to-Speech** — Natural voice output (Piper / XTTS v2)
- **Tool Calling** — Extensible function system for real-world actions
- **100% Offline** — Works without internet after initial setup

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

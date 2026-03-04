# VOX — Voice Operated eXecutive

## Project Overview
Local-first AI voice assistant powered by an RTX 3090. LLM inference, speech
recognition, and voice synthesis all run on your GPU — no cloud AI APIs, no
per-token costs. Optional tools (weather, web) can reach the internet, but
the core voice loop is fully local.

**PUBLIC REPOSITORY** — Never commit secrets, API keys, passwords, or personal data.
CI runs TruffleHog, pattern scanning for secrets, IP addresses, and blocks
.env/.pem/.key files automatically.

## Tech Stack
- **Language**: Python 3.10+
- **Wake Word**: Porcupine (Picovoice) — CPU-only, <100ms detection
- **STT**: Faster-Whisper (CTranslate2 backend) — GPU accelerated
- **LLM**: Ollama (Llama 3.x / Mistral / Qwen) — local, tool calling
- **TTS**: Piper (fast) or XTTS v2 (voice cloning) — GPU accelerated
- **Audio**: sounddevice + numpy for capture/playback
- **GPU**: RTX 3090 (24GB VRAM), CUDA, torch.float16

## Conventions
- Use `pyproject.toml` for project metadata and dependencies
- Use `ruff` for linting
- All config via environment variables (`.env` file, gitignored)
- Models downloaded at runtime to `models/` (gitignored)
- Audio recordings never committed (gitignored)
- **No hardcoded secrets** — use `os.environ` or `python-dotenv`
- Track all bugs/features as GitHub Issues
- Tag releases with semver (v0.1.0, v0.2.0, etc.)

## Directory Structure
```
vox/
├── CLAUDE.md              # Project instructions (this file)
├── README.md              # Public-facing documentation
├── LICENSE                 # MIT
├── SECURITY.md            # Security policy
├── CONTRIBUTING.md        # Contribution guidelines
├── pyproject.toml         # Dependencies and project metadata
├── .env.example           # Template for environment variables
├── .gitignore             # Aggressive secret/model blocking
├── src/vox/
│   ├── __init__.py        # Package init, version
│   ├── __main__.py        # python -m vox
│   ├── cli.py             # CLI entry point (argparse)
│   ├── config.py          # Configuration from env vars
│   ├── audio.py           # Microphone capture, speaker playback
│   ├── wake.py            # Wake word detection (Porcupine)
│   ├── stt.py             # Speech-to-text (Faster-Whisper)
│   ├── llm.py             # LLM interface (Ollama + tool calling)
│   ├── tts.py             # Text-to-speech (Piper / XTTS)
│   ├── tools.py           # Tool definitions for LLM function calling
│   └── pipeline.py        # Main voice loop orchestration
├── models/                # Downloaded models (gitignored)
├── tests/                 # Test suite
└── .github/workflows/     # CI (lint, secret scan)
```

## Security Rules (PUBLIC REPO)
1. **NEVER** commit `.env`, API keys, tokens, or credentials
2. **NEVER** hardcode IP addresses, hostnames, or personal info
3. **ALWAYS** use `.env.example` for config templates (placeholder values only)
4. **ALWAYS** check `git diff` before committing — scan for secrets
5. **ALWAYS** use environment variables for sensitive config
6. Pre-commit hook scans for common secret patterns

## Development Notes
- Porcupine requires a free API key from https://console.picovoice.ai/
- Ollama must be running locally (`ollama serve`)
- Test with `python -m vox` after installing with `pip install -e .`
- VRAM budget: ~2GB STT + ~2GB TTS + ~10-15GB LLM = fits in 24GB

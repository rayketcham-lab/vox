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

---

## Context Window Management — MANDATORY

### Sentinel Protocol
No background watchdog — YOU are the only process. Full protocol: `.claude/agents/sentinel.md`

**Checkpoints**: Before/after tasks, on role switch, after large outputs, every 5 exchanges.
**Heuristics**: <10 tools = green, 10-25 = yellow, 25-40 = orange, 40+ = red compact NOW.
**If asked about Sentinel**: Execute status check, don't explain.

**Session Continuity — NON-NEGOTIABLE**: On compact or exit, ALWAYS update:
1. **MEMORY.md** — task, file paths, what's done/remains, decisions, `## Task Status:` line
2. **CLAUDE.md `## Session State`** (if exists) — current task, files modified, remaining work

Do NOT rewrite CLAUDE.md sections outside `## Session State`. On session start, check MEMORY.md for previous state.

---

## Autonomous Work Mode

Work autonomously. Don't narrate or ask permission — just do the work.

**Decision points**: Use AskUserQuestion with 2-4 concrete options, mark recommended with "(Recommended)".
**Don't ask**: "Should I proceed?" / "Is this okay?" — just do it or present specific options.
**Task Status** in MEMORY.md: `IN_PROGRESS`, `TASK_COMPLETE`, or `BLOCKED`.

---

## Team Agent Architecture

Seven-agent model. Agent details and workflow recipes: see `memory/project-reference.md`.
Agents: Architect, Builder, Tester, SecOps, DevOps, Verifier, Simplifier. Sentinel is embedded discipline, not separate.

**Key rules**:
- Verification-first: Builder → Verifier → then review. Nothing merges unverified.
- SecOps has veto on security. Architect has veto on design/API.
- Builder defers to Tester on test adequacy.

---

## Project Standards

- No warnings in CI. Error handling mandatory. Functions >50 lines → decompose.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `ci:`, `security:`
- No secrets in code. Input validation at trust boundaries. Dependencies audited before adoption.
- New features/bug fixes require tests. Security code requires adversarial tests.

## Conventions
- Use `pyproject.toml` for project metadata and dependencies
- Use `ruff` for linting
- All config via environment variables (`.env` file, gitignored)
- Models downloaded at runtime to `models/` (gitignored)
- Audio recordings never committed (gitignored)
- **No hardcoded secrets** — use `os.environ` or `python-dotenv`
- Track all bugs/features as GitHub Issues
- Tag releases with semver (v0.1.0, v0.2.0, etc.)

## Language Notes

- **Python**: 3.10+, `ruff`, `pathlib`, type hints on public APIs
- **Shell**: quote vars, **NEVER `&&`/`||`/`;` in Bash tool calls** — use separate commands
- Prefer `os.environ` over `python-dotenv` for production; dotenv for dev convenience only

## Directory Structure
```
vox/
├── CLAUDE.md              # Project instructions (this file)
├── README.md              # Public-facing documentation
├── LICENSE                # MIT
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
- Porcupine requires a free API key from console.picovoice.ai
- Ollama must be running locally (`ollama serve`)
- Test with `python -m vox` after installing with `pip install -e .`
- VRAM budget: ~2GB STT + ~2GB TTS + ~10-15GB LLM = fits in 24GB

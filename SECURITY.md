# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability in VOX, please report it responsibly:

1. **Do NOT open a public GitHub issue**
2. Email: security@rayketcham.com
3. Include steps to reproduce and potential impact

## Security Design

VOX is designed with privacy and security as core principles:

- **Fully local**: All processing happens on your machine. No data leaves your network.
- **No telemetry**: VOX does not phone home, track usage, or collect analytics.
- **No cloud APIs**: STT, LLM, and TTS all run locally on your GPU.
- **Secrets management**: All sensitive config via environment variables, never hardcoded.

## For Contributors

- Never commit API keys, tokens, passwords, or personal data
- Use `.env.example` for configuration templates (placeholder values only)
- Pre-commit hooks scan for common secret patterns
- All PRs are reviewed for accidental secret exposure

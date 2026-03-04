# Contributing to VOX

Thanks for your interest in contributing to VOX!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/vox.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install in dev mode: `pip install -e ".[dev]"`
5. Copy `.env.example` to `.env` and fill in your values
6. Create a branch: `git checkout -b feature/your-feature`

## Development

- Run linting: `ruff check src/`
- Run tests: `pytest`
- Run VOX: `python -m vox`

## Pull Requests

1. Keep PRs focused — one feature or fix per PR
2. Update documentation if you change behavior
3. Add tests for new functionality
4. Ensure `ruff check` passes with no errors
5. **NEVER** include secrets, API keys, or personal data in commits

## Code Style

- Python 3.10+ syntax
- `ruff` for formatting and linting
- Type hints encouraged
- Docstrings for public functions

## Reporting Issues

Use GitHub Issues. Include:
- What you expected
- What actually happened
- Steps to reproduce
- Your OS and Python version

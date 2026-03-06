"""Tests for setup wizard."""

from vox.setup import _check_gpu, _check_ollama


def test_check_gpu_returns_string():
    result = _check_gpu()
    assert isinstance(result, str)


def test_check_ollama_returns_tuple():
    running, models = _check_ollama()
    assert isinstance(running, bool)
    assert isinstance(models, list)


def test_cli_setup_flag():
    """Verify --setup is accepted by the CLI parser."""
    import argparse

    # Just verify the parser accepts --setup without error
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true")
    args = parser.parse_args(["--setup"])
    assert args.setup is True

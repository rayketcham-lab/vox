"""Tests for TTS module."""

import numpy as np
import pytest


def test_speak_empty_text():
    from vox.tts import speak
    result = speak("")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_speak_whitespace_only():
    from vox.tts import speak
    result = speak("   ")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_speak_unknown_engine(monkeypatch):
    monkeypatch.setattr("vox.tts.TTS_ENGINE", "nonexistent")
    from vox.tts import speak
    result = speak("Hello world")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_get_sample_rate_piper(monkeypatch):
    monkeypatch.setattr("vox.tts.TTS_ENGINE", "piper")
    from vox.tts import get_sample_rate
    assert get_sample_rate() == 22050


def test_get_sample_rate_xtts(monkeypatch):
    monkeypatch.setattr("vox.tts.TTS_ENGINE", "xtts")
    from vox.tts import get_sample_rate
    assert get_sample_rate() == 24000


def test_piper_not_installed(monkeypatch):
    """Piper should gracefully handle missing binary."""
    monkeypatch.setattr("vox.tts.TTS_ENGINE", "piper")
    # Mock subprocess to raise FileNotFoundError
    import subprocess
    original_run = subprocess.run

    def mock_run(*args, **kwargs):
        raise FileNotFoundError("piper not found")

    monkeypatch.setattr("subprocess.run", mock_run)
    from vox.tts import _speak_piper
    result = _speak_piper("Hello")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_xtts_not_installed(monkeypatch):
    """XTTS should gracefully handle missing TTS library."""
    monkeypatch.setattr("vox.tts._xtts_model", None)
    import sys
    # Temporarily make TTS import fail
    real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def mock_import(name, *args, **kwargs):
        if name == "TTS.api":
            raise ImportError("No module named 'TTS'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)
    from vox.tts import _speak_xtts
    result = _speak_xtts("Hello")
    assert isinstance(result, np.ndarray)
    assert len(result) == 0


def test_unload_xtts(monkeypatch):
    """Unload should work even when no model is loaded."""
    monkeypatch.setattr("vox.tts._xtts_model", None)
    from vox.tts import unload_xtts
    unload_xtts()  # Should not raise

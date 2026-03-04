"""Text-to-speech — pluggable backend (Piper or XTTS)."""

from __future__ import annotations

import numpy as np

from vox.config import TTS_ENGINE


def speak(text: str) -> np.ndarray:
    """Convert text to audio using the configured TTS engine.

    Returns numpy array of audio samples.
    """
    if TTS_ENGINE == "piper":
        return _speak_piper(text)
    elif TTS_ENGINE == "xtts":
        return _speak_xtts(text)
    else:
        print(f"[TTS] Unknown engine '{TTS_ENGINE}', falling back to print.")
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)


def _speak_piper(text: str) -> np.ndarray:
    """TTS via Piper (fast, lightweight)."""
    # TODO: Implement Piper TTS integration
    # For now, just print the response
    print(f"[VOX says]: {text}")
    return np.array([], dtype=np.float32)


def _speak_xtts(text: str) -> np.ndarray:
    """TTS via Coqui XTTS v2 (high quality, voice cloning)."""
    # TODO: Implement XTTS integration
    print(f"[VOX says]: {text}")
    return np.array([], dtype=np.float32)

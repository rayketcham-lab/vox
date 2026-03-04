"""Speech-to-text using Faster-Whisper."""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from vox.config import WHISPER_DEVICE, WHISPER_MODEL

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    """Lazily load the Whisper model."""
    global _model
    if _model is None:
        print(f"[STT] Loading Whisper model '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
        _model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type="float16" if WHISPER_DEVICE == "cuda" else "int8",
        )
        print("[STT] Model loaded.")
    return _model


def transcribe(audio: np.ndarray) -> str:
    """Transcribe audio array to text.

    Args:
        audio: Float32 numpy array at 16kHz mono.

    Returns:
        Transcribed text string.
    """
    model = _get_model()
    segments, info = model.transcribe(audio, beam_size=5, language="en")
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"[STT] Transcribed: {text}")
    return text

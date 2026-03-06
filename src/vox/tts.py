"""Text-to-speech — pluggable backend (Piper or XTTS).

Piper: Fast, CPU-friendly, uses subprocess. Install: pip install piper-tts
XTTS: GPU, high quality, voice cloning. Install: pip install TTS torch
"""

from __future__ import annotations

import logging

import numpy as np

from vox.config import TTS_ENGINE

log = logging.getLogger(__name__)

# Lazy-loaded XTTS model (stays resident in VRAM once loaded)
_xtts_model = None
_xtts_sample_rate = 24000


def speak(text: str) -> np.ndarray:
    """Convert text to audio using the configured TTS engine.

    Returns numpy array of float32 audio samples (mono).
    """
    if not text or not text.strip():
        return np.array([], dtype=np.float32)

    engine = TTS_ENGINE.lower().strip()
    if engine == "piper":
        return _speak_piper(text)
    elif engine == "xtts":
        return _speak_xtts(text)
    else:
        log.warning("Unknown TTS engine '%s', falling back to print.", engine)
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)


def get_sample_rate() -> int:
    """Return the sample rate for the current TTS engine."""
    engine = TTS_ENGINE.lower().strip()
    if engine == "xtts":
        return _xtts_sample_rate
    return 22050  # Piper default


def _speak_piper(text: str) -> np.ndarray:
    """TTS via Piper — fast, lightweight, CPU-friendly.

    Uses the piper CLI which outputs raw WAV to stdout.
    """
    import subprocess

    from vox.config import PIPER_MODEL, PIPER_SPEAKER

    try:
        cmd = ["piper", "--model", PIPER_MODEL, "--output-raw"]
        if PIPER_SPEAKER:
            cmd.extend(["--speaker", str(PIPER_SPEAKER)])

        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            log.error("Piper failed: %s", stderr[:200])
            print(f"[VOX says]: {text}")
            return np.array([], dtype=np.float32)

        # Piper --output-raw outputs 16-bit signed PCM at 22050 Hz mono
        raw_bytes = result.stdout
        if not raw_bytes:
            log.warning("Piper returned empty audio")
            print(f"[VOX says]: {text}")
            return np.array([], dtype=np.float32)

        # Convert 16-bit PCM to float32
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        log.info("Piper generated %d samples (%.1fs)", len(samples), len(samples) / 22050)
        return samples

    except FileNotFoundError:
        log.error("Piper not found — install with: pip install piper-tts")
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)
    except subprocess.TimeoutExpired:
        log.error("Piper timed out after 30s")
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)


def _speak_xtts(text: str) -> np.ndarray:
    """TTS via Coqui XTTS v2 — high quality, voice cloning capable.

    Loads model once and keeps it in GPU VRAM (~2GB).
    """
    global _xtts_model, _xtts_sample_rate

    from vox.config import XTTS_MODEL, XTTS_VOICE_FILE

    try:
        if _xtts_model is None:
            log.info("Loading XTTS model: %s", XTTS_MODEL)
            from TTS.api import TTS
            _xtts_model = TTS(XTTS_MODEL, gpu=True)
            log.info("XTTS model loaded")

        # Generate audio
        kwargs = {"text": text, "language": "en"}
        if XTTS_VOICE_FILE:
            kwargs["speaker_wav"] = XTTS_VOICE_FILE

        wav = _xtts_model.tts(**kwargs)

        if isinstance(wav, list):
            samples = np.array(wav, dtype=np.float32)
        else:
            samples = np.asarray(wav, dtype=np.float32)

        # Normalize if needed
        peak = np.abs(samples).max()
        if peak > 1.0:
            samples = samples / peak

        log.info("XTTS generated %d samples (%.1fs)", len(samples), len(samples) / _xtts_sample_rate)
        return samples

    except ImportError:
        log.error("XTTS not available — install with: pip install TTS torch")
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)
    except Exception as e:
        log.error("XTTS failed: %s", e)
        print(f"[VOX says]: {text}")
        return np.array([], dtype=np.float32)


def unload_xtts() -> None:
    """Unload XTTS model to free VRAM (e.g., before image generation)."""
    global _xtts_model
    if _xtts_model is not None:
        del _xtts_model
        _xtts_model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("XTTS model unloaded")

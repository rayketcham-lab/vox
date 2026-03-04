"""Audio capture and playback using sounddevice."""

from __future__ import annotations

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
CHANNELS = 1


def record_until_silence(
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
    device_index: int = -1,
) -> np.ndarray:
    """Record audio from microphone until silence is detected.

    Returns numpy array of float32 audio samples at 16kHz mono.
    """
    device = device_index if device_index >= 0 else None
    block_size = int(SAMPLE_RATE * 0.1)  # 100ms blocks
    max_blocks = int(max_duration / 0.1)
    silence_blocks = int(silence_duration / 0.1)

    chunks: list[np.ndarray] = []
    silent_count = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32", device=device) as stream:
        for _ in range(max_blocks):
            data, _ = stream.read(block_size)
            chunks.append(data.copy())

            rms = float(np.sqrt(np.mean(data**2)))
            if rms < silence_threshold:
                silent_count += 1
            else:
                silent_count = 0

            if silent_count >= silence_blocks and len(chunks) > silence_blocks:
                break

    return np.concatenate(chunks).flatten()


def play_audio(audio: np.ndarray, sample_rate: int = 22_050, device_index: int = -1) -> None:
    """Play audio array through speakers."""
    device = device_index if device_index >= 0 else None
    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()

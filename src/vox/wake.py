"""Wake word detection using Porcupine."""

from __future__ import annotations

import struct

import numpy as np
import pvporcupine
import sounddevice as sd

from vox.config import PORCUPINE_ACCESS_KEY


def wait_for_wake_word(keyword: str = "hey google", device_index: int = -1) -> None:
    """Block until the wake word is detected.

    Note: Porcupine has built-in keywords. For custom wake words like 'hey vox',
    you'll need to train one at https://console.picovoice.ai/ and pass the .ppn path.
    For now, we use a built-in keyword as placeholder.
    """
    if not PORCUPINE_ACCESS_KEY:
        print("[Wake] No PORCUPINE_ACCESS_KEY set — skipping wake word detection.")
        print("[Wake] Press Enter to simulate wake word...")
        input()
        return

    porcupine = pvporcupine.create(
        access_key=PORCUPINE_ACCESS_KEY,
        keywords=[keyword],
    )

    device = device_index if device_index >= 0 else None
    frame_length = porcupine.frame_length
    sample_rate = porcupine.sample_rate

    print(f'[Wake] Listening for "{keyword}"...')

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
            device=device,
        ) as stream:
            while True:
                data, _ = stream.read(frame_length)
                pcm = struct.unpack_from(f"{frame_length}h", data)
                result = porcupine.process(pcm)
                if result >= 0:
                    print("[Wake] Wake word detected!")
                    return
    finally:
        porcupine.delete()


def wait_for_activation(no_wake: bool = False, device_index: int = -1) -> None:
    """Wait for activation — either wake word or Enter key."""
    if no_wake:
        print("[VOX] Press Enter to speak (or Ctrl+C to quit)...")
        input()
    else:
        wait_for_wake_word(device_index=device_index)

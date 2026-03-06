"""Wake word detection and activation modes.

Supports three listen modes:
- wake: Porcupine wake word detection (default)
- ptt: Push-to-talk via hotkey (F13 or configurable)
- always: Always listening, no activation needed
"""

from __future__ import annotations

import logging
import struct
import threading

import pvporcupine
import sounddevice as sd

from vox.config import LISTEN_MODE, PORCUPINE_ACCESS_KEY, WAKE_SENSITIVITY, WAKE_WORD

log = logging.getLogger(__name__)

# Push-to-talk state
_ptt_active = threading.Event()


def wait_for_wake_word(keyword: str | None = None, device_index: int = -1) -> None:
    """Block until the wake word is detected.

    Uses configurable sensitivity and keyword from config.
    Supports custom .ppn files (path to trained keyword).
    """
    keyword = keyword or WAKE_WORD

    if not PORCUPINE_ACCESS_KEY:
        log.warning("No PORCUPINE_ACCESS_KEY — falling back to Enter key")
        print("[Wake] Press Enter to speak...")
        input()
        return

    # Check if keyword is a path to a custom .ppn file
    from pathlib import Path
    ppn_path = Path(keyword)
    if ppn_path.suffix == ".ppn" and ppn_path.exists():
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[str(ppn_path)],
            sensitivities=[WAKE_SENSITIVITY],
        )
        display_name = ppn_path.stem
    else:
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keywords=[keyword],
            sensitivities=[WAKE_SENSITIVITY],
        )
        display_name = keyword

    device = device_index if device_index >= 0 else None
    frame_length = porcupine.frame_length
    sample_rate = porcupine.sample_rate

    log.info("Listening for '%s' (sensitivity=%.1f)", display_name, WAKE_SENSITIVITY)

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
                    log.info("Wake word detected!")
                    return
    finally:
        porcupine.delete()


def wait_for_ptt() -> None:
    """Block until push-to-talk key is pressed.

    Uses keyboard module if available, otherwise falls back to Enter key.
    """
    try:
        import keyboard
        log.info("Push-to-talk mode — press F13 to speak")
        keyboard.wait("F13")
        return
    except ImportError:
        log.warning("keyboard module not installed — falling back to Enter key for PTT")
        print("[PTT] Press Enter to speak...")
        input()


def wait_for_activation(no_wake: bool = False, device_index: int = -1) -> None:
    """Wait for activation based on configured listen mode.

    Modes:
    - wake: Wait for wake word (default)
    - ptt: Wait for push-to-talk hotkey
    - always: Return immediately (always listening)
    """
    if no_wake:
        print("[VOX] Press Enter to speak (or Ctrl+C to quit)...")
        input()
        return

    mode = LISTEN_MODE.lower().strip()

    if mode == "always":
        log.info("Always-listening mode — recording immediately")
        return

    if mode == "ptt":
        wait_for_ptt()
        return

    # Default: wake word
    wait_for_wake_word(device_index=device_index)

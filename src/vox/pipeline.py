"""Main voice loop orchestration.

Supports two response modes:
  - Text mode: chunks print to terminal as they stream in
  - Voice mode: chunks queue to TTS for real-time speech
"""

from __future__ import annotations

import sys

from vox import __version__
from vox.config import MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX


def _print_config_summary() -> None:
    """Print active configuration so the user knows what's enabled."""
    from vox.config import (
        IMAGE_NSFW_FILTER,
        OLLAMA_MODEL,
        SMTP_HOST,
        TTS_ENGINE,
        VOX_PERSONA_NAME,
        WHISPER_MODEL,
    )

    print(f"  Model:  {OLLAMA_MODEL}")
    print(f"  STT:    Whisper ({WHISPER_MODEL})")
    print(f"  TTS:    {TTS_ENGINE}")
    if VOX_PERSONA_NAME:
        print(f"  Persona: {VOX_PERSONA_NAME}")

    features = []
    if SMTP_HOST:
        features.append("email")
    try:
        import torch  # noqa: F401
        from diffusers import StableDiffusionXLPipeline  # noqa: F401
        features.append("image-gen:SDXL")
        if IMAGE_NSFW_FILTER.lower() == "off":
            features.append("NSFW-filter:off")
    except ImportError:
        try:
            from diffusers import StableDiffusionPipeline  # noqa: F401
            features.append("image-gen:SD1.5")
            if IMAGE_NSFW_FILTER.lower() == "off":
                features.append("NSFW-filter:off")
        except ImportError:
            pass
    if VOX_PERSONA_NAME:
        features.append("persona")
    if features:
        print(f"  Tools:  {', '.join(features)}")


def run(no_wake: bool = False, text_mode: bool = False, model_override: str | None = None) -> None:
    """Main VOX loop: wake -> listen -> think -> speak -> repeat."""
    print("=" * 50)
    print(f"  VOX v{__version__} — Voice Operated eXecutive")
    print("  Local-first AI assistant")
    print("=" * 50)
    _print_config_summary()
    print()

    if text_mode:
        _run_text_mode(model_override)
    else:
        _run_voice_mode(no_wake, model_override)


def _run_text_mode(model_override: str | None) -> None:
    """Text mode with streaming output."""
    from vox.llm import chat

    print("[VOX] Text mode - type your messages below.")
    print("[VOX] Type 'quit' or Ctrl+C to exit.\n")

    def print_chunk(text: str) -> None:
        """Print each chunk as it arrives — no newline, immediate flush."""
        sys.stdout.write(text)
        sys.stdout.flush()

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            sys.stdout.write("VOX: ")
            sys.stdout.flush()
            chat(user_input, model_override=model_override, on_chunk=print_chunk)
            print("\n")
    except (KeyboardInterrupt, EOFError):
        print("\n[VOX] Goodbye.")


def _run_voice_mode(no_wake: bool, model_override: str | None) -> None:
    """Full voice pipeline with streaming TTS."""
    from vox.audio import play_audio, record_until_silence
    from vox.llm import chat
    from vox.stt import transcribe
    from vox.tts import speak
    from vox.wake import wait_for_activation

    print("[VOX] Voice mode - speak after activation.")
    print("[VOX] Ctrl+C to exit.\n")

    try:
        while True:
            # 1. Wait for activation
            wait_for_activation(no_wake=no_wake, device_index=MIC_DEVICE_INDEX)

            # 2. Record speech
            print("[VOX] Listening...")
            audio = record_until_silence(device_index=MIC_DEVICE_INDEX)

            # 3. Transcribe
            text = transcribe(audio)
            if not text.strip():
                print("[VOX] Didn't catch that, try again.")
                continue

            # 4. Think + speak (streamed)
            response = chat(text, model_override=model_override)

            # 5. Full TTS on complete response (until streaming TTS is implemented)
            tts_audio = speak(response)
            if len(tts_audio) > 0:
                play_audio(tts_audio, device_index=SPEAKER_DEVICE_INDEX)

    except KeyboardInterrupt:
        print("\n[VOX] Goodbye.")

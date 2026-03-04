"""Main voice loop orchestration."""

from __future__ import annotations

from vox.config import MIC_DEVICE_INDEX, SPEAKER_DEVICE_INDEX


def run(no_wake: bool = False, text_mode: bool = False, model_override: str | None = None) -> None:
    """Main VOX loop: wake → listen → think → speak → repeat."""
    print("=" * 50)
    print("  VOX — Voice Operated eXecutive")
    print("  Fully local AI assistant")
    print("=" * 50)
    print()

    if text_mode:
        _run_text_mode(model_override)
    else:
        _run_voice_mode(no_wake, model_override)


def _run_text_mode(model_override: str | None) -> None:
    """Text-only mode — no audio hardware needed."""
    from vox.llm import chat

    print("[VOX] Text mode — type your messages below.")
    print("[VOX] Type 'quit' or Ctrl+C to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            response = chat(user_input, model_override=model_override)
            print(f"VOX: {response}\n")
    except (KeyboardInterrupt, EOFError):
        print("\n[VOX] Goodbye.")


def _run_voice_mode(no_wake: bool, model_override: str | None) -> None:
    """Full voice pipeline: wake → listen → think → speak."""
    from vox.audio import play_audio, record_until_silence
    from vox.llm import chat
    from vox.stt import transcribe
    from vox.tts import speak
    from vox.wake import wait_for_activation

    print("[VOX] Voice mode — speak after activation.")
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

            # 4. Think
            response = chat(text, model_override=model_override)

            # 5. Speak
            tts_audio = speak(response)
            if len(tts_audio) > 0:
                play_audio(tts_audio, device_index=SPEAKER_DEVICE_INDEX)

    except KeyboardInterrupt:
        print("\n[VOX] Goodbye.")

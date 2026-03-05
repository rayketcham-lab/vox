"""CLI entry point for VOX."""

import argparse
import logging

from vox import __version__


def main():
    parser = argparse.ArgumentParser(
        prog="vox",
        description="VOX — Voice Operated eXecutive. Local AI voice assistant.",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit")
    parser.add_argument("--no-wake", action="store_true", help="Skip wake word, listen immediately")
    parser.add_argument("--text", action="store_true", help="Text-only mode (no microphone/speaker)")
    parser.add_argument("--model", type=str, default=None, help="Override Ollama model name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging — INFO by default, DEBUG with --debug
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    from vox.pipeline import run

    run(no_wake=args.no_wake, text_mode=args.text, model_override=args.model)


if __name__ == "__main__":
    main()

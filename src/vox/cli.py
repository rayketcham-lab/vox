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
    parser.add_argument("--web", action="store_true", help="Launch web UI instead of voice pipeline")
    parser.add_argument("--port", type=int, default=None, help="Port for web UI (default: 8080)")
    parser.add_argument("--setup", action="store_true", help="Run interactive setup wizard")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging — INFO by default, DEBUG with --debug
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load persona card if configured
    from vox.config import VOX_PERSONA_CARD
    if VOX_PERSONA_CARD:
        from vox.persona import load_card
        load_card(VOX_PERSONA_CARD)

    if args.setup:
        from vox.setup import run_setup
        run_setup()
        return

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    # Auto-suggest setup on first run
    from vox.config import PROJECT_ROOT
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists() and not args.setup:
        print("No .env file found. Run 'vox --setup' to configure VOX.")
        print("Or copy .env.example to .env and edit manually.")
        print()

    if args.web:
        from vox.web import start_server

        start_server(port=args.port)
    else:
        from vox.pipeline import run

        run(no_wake=args.no_wake, text_mode=args.text, model_override=args.model)


if __name__ == "__main__":
    main()

"""First-run setup wizard — interactive configuration for VOX.

Run via `vox --setup`. Auto-detects existing config and skips those steps.
Writes a .env file with all settings.
"""

from __future__ import annotations

import subprocess
import sys


def _color(text: str, code: str) -> str:
    """ANSI color wrapper (skip if not a terminal)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _color(text, "32")


def _yellow(text: str) -> str:
    return _color(text, "33")


def _red(text: str) -> str:
    return _color(text, "31")


def _bold(text: str) -> str:
    return _color(text, "1")


def _prompt(label: str, default: str = "", secret: bool = False) -> str:
    """Prompt user for input with optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        if secret:
            import getpass
            value = getpass.getpass(f"  {label}{suffix}: ")
        else:
            value = input(f"  {label}{suffix}: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value.strip() or default


def _confirm(question: str, default: bool = True) -> bool:
    """Yes/no confirmation."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"  {question} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _check_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is running and list available models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            models = []
            for line in result.stdout.strip().split("\n")[1:]:
                if line.strip():
                    models.append(line.split()[0])
            return True, models
    except Exception:
        return False, []
    return False, []


def _check_gpu() -> str:
    """Detect GPU info."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return ""
    return ""


def run_setup():
    """Run the interactive setup wizard."""
    from vox.config import PROJECT_ROOT

    env_path = PROJECT_ROOT / ".env"
    existing = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                existing[key.strip()] = value.strip()

    config: dict[str, str] = {}

    # Welcome
    print()
    print(_bold("=" * 60))
    print(_bold("  VOX Setup Wizard"))
    print(_bold("=" * 60))
    print()
    print("  This wizard will configure VOX for first use.")
    print("  Press Enter to accept defaults shown in [brackets].")
    print()

    # Step 1: Ollama
    print(_bold("--- Step 1: Ollama (LLM Backend) ---"))
    running, models = _check_ollama()
    if running:
        print(f"  {_green('OK')} Ollama is running with {len(models)} model(s)")
        if models:
            print(f"  Available: {', '.join(models[:10])}")
        default_model = existing.get("OLLAMA_MODEL", "mythomax:13b")
        config["OLLAMA_MODEL"] = _prompt("LLM model", default_model)
    else:
        print(f"  {_red('NOT FOUND')} Ollama is not running")
        print("  Install from https://ollama.ai and run 'ollama serve'")
        config["OLLAMA_MODEL"] = _prompt("LLM model (for later)", "mythomax:13b")
    config["OLLAMA_HOST"] = existing.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    print()

    # Step 2: GPU
    print(_bold("--- Step 2: GPU Detection ---"))
    gpu = _check_gpu()
    if gpu:
        print(f"  {_green('OK')} Detected: {gpu}")
    else:
        print(f"  {_yellow('WARNING')} No NVIDIA GPU detected (image generation will be slow)")
    print()

    # Step 3: User profile
    print(_bold("--- Step 3: User Profile ---"))
    config["USER_EMAIL"] = _prompt("Your email (for 'email me' commands)",
                                   existing.get("USER_EMAIL", ""))
    print()

    # Step 4: Audio
    print(_bold("--- Step 4: Audio Devices ---"))
    print("  Run 'vox --list-devices' to see available devices.")
    if _confirm("Configure audio devices now?", default=False):
        config["MIC_DEVICE_INDEX"] = _prompt("Microphone device index",
                                             existing.get("MIC_DEVICE_INDEX", "0"))
        config["SPEAKER_DEVICE_INDEX"] = _prompt("Speaker device index",
                                                 existing.get("SPEAKER_DEVICE_INDEX", "0"))
    print()

    # Step 5: Wake word
    print(_bold("--- Step 5: Wake Word (Picovoice) ---"))
    print("  Get a free key at https://console.picovoice.ai/")
    pv_key = existing.get("PORCUPINE_ACCESS_KEY", "")
    if pv_key and pv_key != "your_access_key_here":
        print(f"  {_green('OK')} Key already configured")
    else:
        config["PORCUPINE_ACCESS_KEY"] = _prompt("Picovoice API key", "", secret=True)
    print()

    # Step 6: TTS
    print(_bold("--- Step 6: Text-to-Speech ---"))
    tts = _prompt("TTS engine (piper or xtts)", existing.get("TTS_ENGINE", "piper"))
    config["TTS_ENGINE"] = tts
    print()

    # Step 7: Email (optional)
    print(_bold("--- Step 7: Email Relay (optional) ---"))
    if _confirm("Set up email sending?", default=bool(existing.get("SMTP_HOST"))):
        config["SMTP_HOST"] = _prompt("SMTP host", existing.get("SMTP_HOST", ""))
        config["SMTP_PORT"] = _prompt("SMTP port", existing.get("SMTP_PORT", "587"))
        config["SMTP_USER"] = _prompt("SMTP username", existing.get("SMTP_USER", ""))
        config["SMTP_PASSWORD"] = _prompt("SMTP password", "", secret=True)
        config["SMTP_FROM"] = _prompt("From address",
                                      existing.get("SMTP_FROM", config.get("USER_EMAIL", "")))
    print()

    # Step 8: Image generation
    print(_bold("--- Step 8: Image Generation (optional) ---"))
    if gpu and _confirm("Enable image generation?", default=True):
        config["IMAGE_MODEL"] = _prompt("SFW model",
                                        existing.get("IMAGE_MODEL",
                                                     "stabilityai/stable-diffusion-xl-base-1.0"))
        config["IMAGE_NSFW_FILTER"] = _prompt("NSFW filter (on/off)",
                                              existing.get("IMAGE_NSFW_FILTER", "on"))
    print()

    # Step 9: Persona
    print(_bold("--- Step 9: Persona (optional) ---"))
    if _confirm("Give VOX a persona/character?", default=bool(existing.get("VOX_PERSONA_NAME"))):
        config["VOX_PERSONA_NAME"] = _prompt("Persona name",
                                             existing.get("VOX_PERSONA_NAME", "Luna"))
        config["VOX_PERSONA_DESCRIPTION"] = _prompt(
            "Appearance description",
            existing.get("VOX_PERSONA_DESCRIPTION",
                         "a young woman with long dark hair, green eyes"),
        )
    print()

    # Step 10: Write .env
    print(_bold("--- Configuration Summary ---"))
    for key, value in config.items():
        display = "***" if "PASSWORD" in key or "KEY" in key else value
        if value:
            print(f"  {key}={display}")
    print()

    if _confirm("Write configuration to .env?"):
        # Merge with existing
        final = {**existing, **config}
        # Remove empty values
        final = {k: v for k, v in final.items() if v}
        lines = [f"{k}={v}" for k, v in final.items()]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n  {_green('OK')} Configuration saved to {env_path}")
    else:
        print(f"\n  {_yellow('SKIPPED')} No changes written")

    print()
    print(_bold("=" * 60))
    print(_bold(f"  {_green('Your VOX is ready!')}"))
    print(_bold("=" * 60))
    print()
    print("  Start with:  vox --web      (web UI)")
    print("               vox --text     (text mode)")
    print("               vox            (voice mode)")
    print()

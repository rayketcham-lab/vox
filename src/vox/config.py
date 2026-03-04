"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Wake word
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY", "")
WAKE_WORD = "hey vox"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

# Audio
MIC_DEVICE_INDEX = int(os.environ.get("MIC_DEVICE_INDEX", -1))
SPEAKER_DEVICE_INDEX = int(os.environ.get("SPEAKER_DEVICE_INDEX", -1))

# STT
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")

# TTS
TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper")

# Home Assistant (optional)
HASS_URL = os.environ.get("HASS_URL", "")
HASS_TOKEN = os.environ.get("HASS_TOKEN", "")

# System prompt
SYSTEM_PROMPT = """You are VOX, a helpful local AI assistant. You run entirely on the user's hardware.
Be concise and direct. Respond in 1-3 sentences unless asked for detail.
You have access to tools for controlling the local environment."""

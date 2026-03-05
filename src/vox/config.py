"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
DOWNLOADS_DIR = PROJECT_ROOT / "downloads"
DOWNLOADS_DIR.mkdir(exist_ok=True)

# HuggingFace cache — redirect to project drive to avoid filling OS drive
HF_HOME = os.environ.get("HF_HOME", str(PROJECT_ROOT / "models" / "huggingface"))
os.environ["HF_HOME"] = HF_HOME
Path(HF_HOME).mkdir(parents=True, exist_ok=True)

# Wake word
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY", "")
WAKE_WORD = "hey vox"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mythomax:13b")

# Audio
MIC_DEVICE_INDEX = int(os.environ.get("MIC_DEVICE_INDEX", -1))
SPEAKER_DEVICE_INDEX = int(os.environ.get("SPEAKER_DEVICE_INDEX", -1))

# STT
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda")

# TTS
TTS_ENGINE = os.environ.get("TTS_ENGINE", "piper")

# Email (optional — SMTP for sending emails via tools)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
SMTP_FROM = os.environ.get("SMTP_FROM", "")
USER_EMAIL = os.environ.get("USER_EMAIL", "")  # default "email me" recipient

# Image generation (optional — requires diffusers)
# Dual-model setup: SDXL base for SFW, Juggernaut for NSFW/persona
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGE_MODEL_NSFW = os.environ.get("IMAGE_MODEL_NSFW", "RunDiffusion/Juggernaut-X-v10")
IMAGE_NSFW_FILTER = os.environ.get("IMAGE_NSFW_FILTER", "on")
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "30"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "1024"))
IMAGE_NEGATIVE_PROMPT = os.environ.get(
    "IMAGE_NEGATIVE_PROMPT",
    "deformed, bad anatomy, extra limbs, mutated hands, poorly drawn face, "
    "blurry, low quality, watermark, text, signature, cropped, "
    "cgi, 3d render, cartoon, anime, illustration, painting, drawing, "
    "airbrushed, plastic skin, smooth skin, doll-like, overly perfect, "
    "symmetrical face, uncanny valley",
)

# Persona (optional — gives VOX a visual identity for selfie generation)
VOX_PERSONA_NAME = os.environ.get("VOX_PERSONA_NAME", "")
VOX_PERSONA_DESCRIPTION = os.environ.get("VOX_PERSONA_DESCRIPTION", "")
VOX_PERSONA_STYLE = os.environ.get(
    "VOX_PERSONA_STYLE",
    "photorealistic, shot on Canon EOS R5 85mm f/1.4, natural lighting, "
    "shallow depth of field, visible skin pores, freckles, beauty marks, "
    "natural skin texture, film grain, candid pose, raw photo, 8k",
)

# Persona card (optional — YAML character card for rich personality)
VOX_PERSONA_CARD = os.environ.get("VOX_PERSONA_CARD", "")

# Vision model (for analyzing uploaded images — must support multimodal)
VISION_MODEL = os.environ.get("VISION_MODEL", "llava")

# Web UI
WEB_HOST = os.environ.get("WEB_HOST", "0.0.0.0")  # noqa: S104 — intentional LAN binding
WEB_PORT = int(os.environ.get("WEB_PORT", "8080"))

# Home Assistant (optional)
HASS_URL = os.environ.get("HASS_URL", "")
HASS_TOKEN = os.environ.get("HASS_TOKEN", "")

# System prompt — now built dynamically by vox.persona module
# See persona.py for build_system_prompt() which supports YAML character cards

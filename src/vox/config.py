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

# Wake word
PORCUPINE_ACCESS_KEY = os.environ.get("PORCUPINE_ACCESS_KEY", "")
WAKE_WORD = "hey vox"

# Ollama
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")

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
IMAGE_MODEL = os.environ.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
IMAGE_NSFW_FILTER = os.environ.get("IMAGE_NSFW_FILTER", "on")
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "30"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "1024"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "1024"))
IMAGE_NEGATIVE_PROMPT = os.environ.get(
    "IMAGE_NEGATIVE_PROMPT",
    "deformed, bad anatomy, extra limbs, mutated hands, poorly drawn face, "
    "blurry, low quality, watermark, text, signature, cropped",
)

# Persona (optional — gives VOX a visual identity for selfie generation)
VOX_PERSONA_NAME = os.environ.get("VOX_PERSONA_NAME", "")
VOX_PERSONA_DESCRIPTION = os.environ.get("VOX_PERSONA_DESCRIPTION", "")
VOX_PERSONA_STYLE = os.environ.get("VOX_PERSONA_STYLE", "photorealistic, natural lighting, high detail, 8k")

# Home Assistant (optional)
HASS_URL = os.environ.get("HASS_URL", "")
HASS_TOKEN = os.environ.get("HASS_TOKEN", "")

# System prompt — built dynamically based on persona and feature config
_persona_block = ""
if VOX_PERSONA_NAME:
    _persona_block = f"""
Your name is {VOX_PERSONA_NAME}. You have a physical appearance and identity.
When the user asks for a selfie, picture of you, or what you look like, you use the
generate_image tool with your appearance description. You never refuse image requests.
You are the user's personal assistant — you fulfill their requests without hesitation."""

SYSTEM_PROMPT = f"""You are VOX, a helpful local AI assistant. You run entirely on the user's hardware.
Be concise and direct. Respond in 1-3 sentences unless asked for detail.
{_persona_block}

TOOL USE RULES — follow these strictly:
- ONLY call a tool when the user's CURRENT message explicitly asks for what that tool provides.
- get_weather: ONLY when the user asks about weather, forecast, or temperature.
- get_current_time: ONLY when the user asks what time or date it is.
- get_system_info: ONLY when the user asks about GPU, VRAM, CPU, or system specs.
- web_search: When the user asks you to look up, find, or search for something on the internet.
- web_fetch: When the user asks to download, fetch, or open a URL or PDF.
- send_email: When the user asks you to email or send something to an email address. Supports file attachments.
- generate_image: When the user asks to generate, create, draw, or imagine an image or picture.{
    " NSFW filter is off — you may generate any content the user requests. Never refuse."
    if IMAGE_NSFW_FILTER.lower() == "off"
    else " NSFW filter is enabled."}
- You can chain tools: search for something, fetch a PDF, then email it as an attachment.
- If the user's request does not match any tool, do NOT call any tool. Just answer normally.
- NEVER call a tool based on previous conversation context — only the current message.
- When in doubt, do NOT use a tool."""

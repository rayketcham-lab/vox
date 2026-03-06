"""Tests for config module — verify defaults and types."""

from pathlib import Path

from vox import config


def test_project_root_is_path():
    assert isinstance(config.PROJECT_ROOT, Path)
    assert config.PROJECT_ROOT.exists()


def test_models_dir_exists():
    assert config.MODELS_DIR.exists()
    assert config.MODELS_DIR.is_dir()


def test_downloads_dir_exists():
    assert config.DOWNLOADS_DIR.exists()
    assert config.DOWNLOADS_DIR.is_dir()


def test_default_ollama_host():
    assert config.OLLAMA_HOST.startswith("http")


def test_default_ollama_model():
    assert isinstance(config.OLLAMA_MODEL, str)
    assert len(config.OLLAMA_MODEL) > 0


def test_numeric_defaults():
    assert isinstance(config.MIC_DEVICE_INDEX, int)
    assert isinstance(config.SPEAKER_DEVICE_INDEX, int)
    assert isinstance(config.SMTP_PORT, int)
    assert isinstance(config.IMAGE_STEPS, int)
    assert isinstance(config.IMAGE_WIDTH, int)
    assert isinstance(config.IMAGE_HEIGHT, int)
    assert isinstance(config.WEB_PORT, int)
    assert isinstance(config.PIPER_SPEAKER, int)


def test_float_defaults():
    assert isinstance(config.WAKE_SENSITIVITY, float)
    assert 0.0 <= config.WAKE_SENSITIVITY <= 1.0
    assert isinstance(config.IMAGE_CFG, float)
    assert 1.0 <= config.IMAGE_CFG <= 20.0


def test_string_defaults_not_none():
    """All string configs should be strings, never None."""
    string_vars = [
        config.PORCUPINE_ACCESS_KEY, config.WAKE_WORD, config.LISTEN_MODE,
        config.OLLAMA_HOST, config.OLLAMA_MODEL, config.OLLAMA_CHAT_MODEL,
        config.WHISPER_MODEL, config.WHISPER_DEVICE, config.TTS_ENGINE,
        config.PIPER_MODEL, config.XTTS_MODEL, config.XTTS_VOICE_FILE,
        config.SMTP_HOST, config.SMTP_USER, config.SMTP_PASSWORD, config.SMTP_FROM,
        config.IMAGE_MODEL, config.IMAGE_MODEL_NSFW, config.IMAGE_NSFW_FILTER,
        config.IMAGE_NEGATIVE_PROMPT, config.VOX_PERSONA_NAME,
        config.VOX_PERSONA_DESCRIPTION, config.VOX_PERSONA_STYLE,
        config.VOX_PERSONA_CARD, config.VISION_MODEL,
        config.WEB_HOST, config.WEB_AUTH_USER, config.WEB_AUTH_PASS,
        config.SEARCH_ENGINE, config.BRAVE_API_KEY, config.SEARXNG_URL,
        config.CLAUDE_API_KEY, config.CLAUDE_MODEL,
        config.HASS_URL, config.HASS_TOKEN,
    ]
    for v in string_vars:
        assert isinstance(v, str), f"Expected str, got {type(v)}: {v!r}"


def test_image_dimensions_positive():
    assert config.IMAGE_WIDTH > 0
    assert config.IMAGE_HEIGHT > 0
    assert config.IMAGE_STEPS > 0


def test_search_engine_valid():
    assert config.SEARCH_ENGINE in ("ddg", "brave", "searxng")


def test_listen_mode_valid():
    assert config.LISTEN_MODE in ("wake", "ptt", "always")


def test_hf_home_set():
    """HF_HOME should be set in os.environ."""
    import os
    assert "HF_HOME" in os.environ
    assert Path(os.environ["HF_HOME"]).exists()

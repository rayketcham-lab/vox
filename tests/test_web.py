"""Tests for web.py utility functions — image detection, path stripping, origin check."""

import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from vox.web import _check_ws_origin, _find_generated_images  # noqa: E402

# ---------------------------------------------------------------------------
# _find_generated_images
# ---------------------------------------------------------------------------

def test_find_generated_images_basic():
    history = [
        {"role": "user", "content": "take a selfie"},
        {"role": "tool", "content": "Image saved to vox_image_20260305.png"},
    ]
    result = _find_generated_images(history, 0)
    assert result == ["vox_image_20260305.png"]


def test_find_generated_images_multiple():
    history = [
        {"role": "tool", "content": "saved vox_image_001.png"},
        {"role": "tool", "content": "map at vox_map_nyc.png"},
    ]
    result = _find_generated_images(history, 0)
    assert "vox_image_001.png" in result
    assert "vox_map_nyc.png" in result


def test_find_generated_images_before_len():
    """Only entries after before_len should be scanned."""
    history = [
        {"role": "tool", "content": "old vox_image_old.png"},
        {"role": "tool", "content": "new vox_image_new.png"},
    ]
    result = _find_generated_images(history, 1)
    assert result == ["vox_image_new.png"]


def test_find_generated_images_no_match():
    history = [
        {"role": "tool", "content": "Weather is 72F and sunny"},
    ]
    result = _find_generated_images(history, 0)
    assert result == []


def test_find_generated_images_skips_non_tool():
    history = [
        {"role": "assistant", "content": "Here's vox_image_fake.png"},
    ]
    result = _find_generated_images(history, 0)
    assert result == []


def test_find_generated_images_strips_trailing_comma():
    history = [
        {"role": "tool", "content": "saved vox_image_abc.png, done"},
    ]
    result = _find_generated_images(history, 0)
    assert result == ["vox_image_abc.png"]


# ---------------------------------------------------------------------------
# Path stripping regex patterns (same patterns used in web.py _handle_chat)
# ---------------------------------------------------------------------------

def _strip_paths(text: str) -> str:
    """Apply the same path-stripping regexes used in web.py."""
    text = re.sub(r"[A-Z]:\\[\w\\]+\.\w+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:saved?\s+(?:it\s+)?to|stored\s+(?:it\s+)?(?:at|in)?)\s+\S+\.png\b",
        "", text, flags=re.IGNORECASE,
    )
    text = re.sub(r"\bvox_(?:image|map)_\S+\.png\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def test_strip_drive_letter_path():
    assert "file" not in _strip_paths("I saved your photo to D:\\downloads\\file.png")


def test_strip_vox_image_filename():
    result = _strip_paths("Here's your selfie vox_image_20260305_123456.png enjoy!")
    assert "vox_image" not in result
    assert "enjoy" in result


def test_strip_saved_to_path():
    result = _strip_paths("Photo saved to /downloads/vox_image_test.png for you")
    assert "vox_image" not in result


def test_strip_preserves_normal_text():
    text = "The weather is 72 degrees and sunny!"
    assert _strip_paths(text) == text


def test_strip_multiple_paths():
    text = "Generated vox_image_a.png and vox_map_b.png for you"
    result = _strip_paths(text)
    assert "vox_image" not in result
    assert "vox_map" not in result


# ---------------------------------------------------------------------------
# WebSocket origin validation
# ---------------------------------------------------------------------------

def _mock_ws(origin: str = "", host: str = "localhost:8080"):
    ws = MagicMock()
    headers = {"host": host}
    if origin:
        headers["origin"] = origin
    ws.headers = headers
    return ws


def test_ws_origin_no_origin_header():
    """Non-browser clients without Origin should be allowed."""
    assert _check_ws_origin(_mock_ws(origin="", host="localhost:8080"))


def test_ws_origin_same_host():
    ws = _mock_ws(origin="http://192.168.1.50:8080", host="192.168.1.50:8080")
    assert _check_ws_origin(ws)


def test_ws_origin_localhost_variants():
    ws = _mock_ws(origin="http://127.0.0.1:8080", host="localhost:8080")
    assert _check_ws_origin(ws)


def test_ws_origin_cross_site_rejected():
    ws = _mock_ws(origin="http://evil.com", host="192.168.1.50:8080")
    assert not _check_ws_origin(ws)


def test_ws_origin_different_port_same_host():
    ws = _mock_ws(origin="http://192.168.1.50:3000", host="192.168.1.50:8080")
    assert _check_ws_origin(ws)

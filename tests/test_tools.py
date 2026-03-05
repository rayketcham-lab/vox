"""Tests for tool execution, intent detection, and validation."""

import pytest

from vox.tools import (
    _clean_ddg_url,
    _extract_image_prompt,
    detect_all_intents,
    detect_intent,
    execute_tool,
    validate_tool_call,
)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def test_get_current_time():
    result = execute_tool("get_current_time", {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result


def test_get_system_info():
    result = execute_tool("get_system_info", {})
    assert "OS:" in result
    assert "Python:" in result


def test_send_email_missing_recipient():
    """Email tool should fail gracefully when no recipient is provided."""
    result = execute_tool("send_email", {"to": "", "subject": "Test", "body": "Hi"})
    assert "No recipient" in result


# ---------------------------------------------------------------------------
# Intent detection — should match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_tool", [
    # Weather
    ("what's the weather today?", "get_weather"),
    ("give me the forecast for this week", "get_weather"),
    ("is it going to rain tomorrow?", "get_weather"),
    ("how's the temperature outside?", "get_weather"),
    ("is it sunny right now?", "get_weather"),
    ("will it snow this weekend?", "get_weather"),
    # Time
    ("what time is it?", "get_current_time"),
    ("what's the current time?", "get_current_time"),
    # System
    ("tell me the system info", "get_system_info"),
    ("how much vram do I have?", "get_system_info"),
    ("what gpu am I running?", "get_system_info"),
    ("check my memory usage", "get_system_info"),
    # Web search
    ("search for 2007 Chevy Tahoe wiring diagram", "web_search"),
    ("look up the specs for an RTX 4090", "web_search"),
    ("find me schematics for the electrical system on a 2007 Chevy Tahoe", "web_search"),
    ("google best pizza near me", "web_search"),
    # Email
    ("email me at user@example.com", "send_email"),
    ("can you email the results to test@example.com", "send_email"),
    # Image generation
    ("generate an image of a sunset", "generate_image"),
    ("create a picture of a dog", "generate_image"),
    ("draw me a landscape", "generate_image"),
    ("paint me a portrait", "generate_image"),
    ("imagine a futuristic city", "generate_image"),
    ("make an illustration of a cat", "generate_image"),
])
def test_intent_detection_matches(text, expected_tool):
    intent = detect_intent(text)
    assert intent is not None, f"Expected intent for: {text}"
    assert intent.tool_name == expected_tool


# ---------------------------------------------------------------------------
# Intent detection — should NOT match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "tell me about the history of Python programming",
    "how do I fix a leaking faucet?",
    "what's the best recipe for banana bread?",
    "explain quantum computing in simple terms",
    "help me debug this JavaScript error",
    "my car's engine is overheating, what should I do?",
    "write me a poem about the ocean",
    "how do I change the oil in my truck?",
    "what are the system requirements for Cyberpunk 2077?",
    "hello, how are you doing today?",
    "thanks for the help",
    "tell me about image formats",
    "how do digital images work?",
    "draw a conclusion from this data",
])
def test_intent_detection_no_false_positives(text):
    intent = detect_intent(text)
    assert intent is None, f"Unexpected intent '{intent.tool_name}' for: {text}"


# ---------------------------------------------------------------------------
# Tool call validation — blocks spurious LLM tool calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool,text", [
    ("get_weather", "what's the weather in Dallas?"),
    ("get_weather", "is it raining outside?"),
    ("get_weather", "will it be hot tomorrow?"),
    ("get_weather", "give me the forecast"),
    ("get_current_time", "what time is it right now?"),
    ("get_current_time", "what's today's date?"),
    ("get_system_info", "how much vram is free?"),
    ("get_system_info", "tell me my gpu stats"),
    ("get_system_info", "show me the hardware info"),
    ("web_search", "search for Python tutorials"),
    ("web_search", "look up the weather in Paris"),
    ("web_search", "find me a recipe for tacos"),
    ("send_email", "email this to ray@example.com"),
    ("send_email", "send the results to test@test.com"),
    ("generate_image", "create an image of a sunset"),
    ("generate_image", "draw me a picture of a dog"),
    ("generate_image", "paint me something beautiful"),
])
def test_validate_tool_call_allows_relevant(tool, text):
    assert validate_tool_call(tool, text) is True


@pytest.mark.parametrize("tool,text", [
    # The original bug: LLM called get_weather for a car schematic question
    ("get_weather", "find me schematics for the electrical system on a 2007 Chevy Tahoe PPV"),
    ("get_weather", "help me write a Python function"),
    ("get_weather", "what's the capital of France?"),
    ("get_weather", "tell me about the stock market"),
    ("get_current_time", "how do I bake a cake?"),
    ("get_current_time", "explain machine learning"),
    ("get_system_info", "what are the system requirements for this game?"),
    ("get_system_info", "how much memory does Chrome use?"),
    # New tools — wrong context
    ("web_search", "tell me about the history of Python"),
    ("web_search", "how do I fix a leaking faucet?"),
    ("send_email", "tell me about email protocols"),
    ("send_email", "how does SMTP work?"),
    ("generate_image", "tell me about image compression"),
    ("generate_image", "how does JPEG encoding work?"),
])
def test_validate_tool_call_blocks_irrelevant(tool, text):
    assert validate_tool_call(tool, text) is False


def test_validate_unknown_tool_passes_through():
    """Unknown tools should pass validation (execute_tool handles the error)."""
    assert validate_tool_call("nonexistent_tool", "anything") is True


# ---------------------------------------------------------------------------
# Arg extraction — web_search gets a real query, send_email gets an address
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_query_contains", [
    (
        "find me schematics for the electrical system on a 2007 Chevy Tahoe",
        "schematics for the electrical system on a 2007 Chevy Tahoe",
    ),
    (
        "search for RTX 4090 specs",
        "RTX 4090 specs",
    ),
    (
        "Can you look up Python asyncio tutorial",
        "look up Python asyncio tutorial",
    ),
])
def test_web_search_extracts_query(text, expected_query_contains):
    intent = detect_intent(text)
    assert intent is not None
    assert intent.tool_name == "web_search"
    query = intent.args.get("query", "")
    assert len(query) > 5, f"Query too short: '{query}'"
    assert expected_query_contains.lower() in query.lower(), (
        f"Expected '{expected_query_contains}' in query '{query}'"
    )


@pytest.mark.parametrize("text,expected_email", [
    ("email me at ray@example.com", "ray@example.com"),
    ("can you email the results to test@test.org", "test@test.org"),
])
def test_send_email_extracts_address(text, expected_email):
    intent = detect_intent(text)
    assert intent is not None
    assert intent.tool_name == "send_email"
    assert intent.args.get("to") == expected_email


def test_combined_search_and_email_detects_search_first():
    """When user asks to search AND email, search should fire first (it comes first in patterns)."""
    text = "find me schematics for a Chevy Tahoe and email them to ray@test.com"
    intent = detect_intent(text)
    assert intent is not None
    assert intent.tool_name == "web_search"
    query = intent.args.get("query", "")
    # Query should NOT include the email tail
    assert "@" not in query, f"Email leaked into query: '{query}'"
    assert len(query) > 5


# ---------------------------------------------------------------------------
# Multi-intent detection (tool chaining)
# ---------------------------------------------------------------------------

def test_detect_all_intents_search_and_email():
    """Should detect both web_search and send_email in a combined request."""
    text = "find me schematics for a Chevy Tahoe and email them to ray@test.com"
    intents = detect_all_intents(text)
    assert len(intents) == 2
    assert intents[0].tool_name == "web_search"
    assert intents[1].tool_name == "send_email"
    assert intents[1].args.get("to") == "ray@test.com"
    # Search query should not include email tail
    assert "@" not in intents[0].args.get("query", "")


def test_detect_all_intents_single_tool():
    """Single intent should return a list with one element."""
    text = "what's the weather today?"
    intents = detect_all_intents(text)
    assert len(intents) == 1
    assert intents[0].tool_name == "get_weather"


def test_detect_all_intents_no_match():
    """Non-tool text should return empty list."""
    intents = detect_all_intents("tell me about Python programming")
    assert intents == []


def test_detect_all_intents_no_duplicates():
    """Should not return the same tool twice."""
    text = "search for pizza recipes and also find me Italian restaurants"
    intents = detect_all_intents(text)
    tool_names = [i.tool_name for i in intents]
    assert tool_names.count("web_search") == 1


@pytest.mark.parametrize("text,expected_tools", [
    (
        "look up RTX 4090 specs and email them to ray@test.com",
        ["web_search", "send_email"],
    ),
    (
        "search for wiring diagrams and email me at test@example.com",
        ["web_search", "send_email"],
    ),
    (
        "what's the weather and what time is it?",
        ["get_weather", "get_current_time"],
    ),
])
def test_detect_all_intents_chaining_cases(text, expected_tools):
    intents = detect_all_intents(text)
    tool_names = [i.tool_name for i in intents]
    assert tool_names == expected_tools


# ---------------------------------------------------------------------------
# send_email attachment support
# ---------------------------------------------------------------------------

def test_send_email_empty_attachments_list():
    """Empty attachments list should not break email (no regression)."""
    result = execute_tool("send_email", {
        "to": "test@example.com", "subject": "Test", "body": "Hi",
        "attachments": [],
    })
    # Without SMTP_HOST configured, this returns the config error — that's fine,
    # we just need to confirm it doesn't crash on empty attachments.
    assert isinstance(result, str)
    assert "No recipient" not in result  # recipient was provided


def test_send_email_empty_string_attachments():
    """Empty string for attachments (backward compat) should not crash."""
    result = execute_tool("send_email", {
        "to": "test@example.com", "subject": "Test", "body": "Hi",
        "attachments": "",
    })
    assert isinstance(result, str)
    assert "No recipient" not in result


def test_send_email_nonexistent_attachment(tmp_path):
    """Nonexistent attachment file path should produce a warning."""
    # We need SMTP to be configured for this to reach the attachment logic.
    # Mock SMTP to avoid needing a real server.
    from unittest.mock import MagicMock, patch

    fake_path = str(tmp_path / "does_not_exist.pdf")

    with patch("vox.config.SMTP_HOST", "localhost"), \
         patch("vox.config.SMTP_PORT", 25), \
         patch("vox.config.SMTP_USER", ""), \
         patch("vox.config.SMTP_PASSWORD", ""), \
         patch("vox.config.SMTP_FROM", "vox@test.com"), \
         patch("smtplib.SMTP") as mock_smtp:
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        mock_server.has_extn.return_value = False

        result = execute_tool("send_email", {
            "to": "test@example.com",
            "subject": "Test",
            "body": "Hi",
            "attachments": [fake_path],
        })

    assert "Attachment not found" in result or "Email sent" in result


# ---------------------------------------------------------------------------
# DDG redirect URL cleaning
# ---------------------------------------------------------------------------

def test_clean_ddg_url_redirect():
    """DDG redirect URL should be unwrapped to the real URL."""
    ddg = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage%3Fq%3D1&rut=abc123"
    assert _clean_ddg_url(ddg) == "https://example.com/page?q=1"


def test_clean_ddg_url_normal():
    """A normal URL should be returned unchanged."""
    url = "https://example.com/page"
    assert _clean_ddg_url(url) == url


def test_clean_ddg_url_empty():
    """Empty string should be returned as-is."""
    assert _clean_ddg_url("") == ""


def test_clean_ddg_url_missing_uddg():
    """DDG redirect URL without uddg param should return original."""
    url = "//duckduckgo.com/l/?rut=abc123"
    assert _clean_ddg_url(url) == url


# ---------------------------------------------------------------------------
# web_fetch — intent detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_tool", [
    ("download the PDF from https://example.com/doc.pdf", "web_fetch"),
    ("fetch this page: https://example.com", "web_fetch"),
    ("grab the PDF at https://example.com/report.pdf", "web_fetch"),
    ("open the website https://example.com", "web_fetch"),
    ("get the page at https://example.com/info", "web_fetch"),
])
def test_web_fetch_intent_matches(text, expected_tool):
    intent = detect_intent(text)
    assert intent is not None, f"Expected intent for: {text}"
    assert intent.tool_name == expected_tool


def test_web_fetch_no_false_positive():
    """Should NOT match 'can you help me download Python?' — no pdf/page/url/link/site keyword."""
    intent = detect_intent("can you help me download Python?")
    # 'download' alone without pdf/page/url/link/site/website should not match web_fetch
    # and there's no URL in the text, so neither pattern should fire
    assert intent is None or intent.tool_name != "web_fetch", (
        f"Unexpected web_fetch intent for generic download request"
    )


# ---------------------------------------------------------------------------
# web_fetch — URL extraction
# ---------------------------------------------------------------------------

def test_extract_url_from_text():
    from vox.tools import _extract_url

    assert _extract_url("check out https://example.com/page") == "https://example.com/page"
    assert _extract_url("visit http://test.org/foo?bar=1") == "http://test.org/foo?bar=1"
    assert _extract_url("no url here") == ""
    # Trailing punctuation should be stripped
    assert _extract_url("see https://example.com.") == "https://example.com"


# ---------------------------------------------------------------------------
# web_fetch — validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "download the PDF from https://example.com/doc.pdf",
    "fetch the page at https://example.com",
    "grab the website content",
    "get the link for me",
    "https://example.com",
])
def test_validate_web_fetch_allows(text):
    assert validate_tool_call("web_fetch", text) is True


@pytest.mark.parametrize("text", [
    "tell me about the history of Python",
    "how do I bake a cake?",
    "explain quantum computing",
])
def test_validate_web_fetch_blocks(text):
    assert validate_tool_call("web_fetch", text) is False


# ---------------------------------------------------------------------------
# web_fetch — execution edge cases
# ---------------------------------------------------------------------------

def test_web_fetch_empty_url():
    result = execute_tool("web_fetch", {"url": ""})
    assert "No URL provided" in result


def test_web_fetch_invalid_url():
    result = execute_tool("web_fetch", {"url": "not-a-url"})
    assert "Invalid URL" in result


# ---------------------------------------------------------------------------
# generate_image — execution edge cases
# ---------------------------------------------------------------------------

def test_generate_image_no_prompt():
    """Empty prompt should return an error message."""
    result = execute_tool("generate_image", {"prompt": ""})
    assert "No" in result and "prompt" in result.lower()


# ---------------------------------------------------------------------------
# generate_image — prompt extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected_fragment", [
    ("generate an image of a sunset over the ocean", "sunset over the ocean"),
    ("draw me a picture of a happy cat", "happy cat"),
    # Conversational tails should be stripped
    ("draw me a naked brunette, but we should be able to utilize the 3090 to draw it", "naked brunette"),
    ("paint me a landscape, however you can make it any style", "landscape"),
])
def test_extract_image_prompt(text, expected_fragment):
    prompt = _extract_image_prompt(text)
    assert expected_fragment in prompt.lower(), (
        f"Expected '{expected_fragment}' in extracted prompt '{prompt}'"
    )


# ---------------------------------------------------------------------------
# generate_image — chaining with other tools
# ---------------------------------------------------------------------------

def test_generate_image_and_email_chaining():
    """Should detect both generate_image and send_email in a combined request."""
    text = "draw me a sunset and email it to user@example.com"
    intents = detect_all_intents(text)
    tool_names = [i.tool_name for i in intents]
    assert "generate_image" in tool_names
    assert "send_email" in tool_names

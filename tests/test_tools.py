"""Tests for tool execution, intent detection, and validation."""

import pytest

from vox.tools import (
    _build_persona_prompt,
    _clean_ddg_url,
    _extract_image_prompt,
    _is_selfie_request,
    _should_use_nsfw_model,
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
    ("what is the time", "get_current_time"),
    ("what is the date today", "get_current_time"),
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
    ("research quantum computing", "web_search"),
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
    ("send_email", "email me the report"),
    ("send_email", "send it to me"),
    ("generate_image", "create an image of a sunset"),
    ("generate_image", "draw me a picture of a dog"),
    ("generate_image", "paint me something beautiful"),
    ("take_screenshot", "take a screenshot"),
    ("take_screenshot", "capture my screen"),
    ("take_screenshot", "what's on my screen"),
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
        "Python asyncio tutorial",
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


def test_email_me_without_address(monkeypatch):
    """'email me' (no address) should detect send_email and use USER_EMAIL fallback."""
    monkeypatch.setattr("vox.config.USER_EMAIL", "user@example.com")
    intent = detect_intent("email me the results")
    assert intent is not None
    assert intent.tool_name == "send_email"
    assert intent.args.get("to") == "user@example.com"


def test_email_me_no_config():
    """'email me' with no USER_EMAIL set should still detect intent (empty 'to')."""
    # USER_EMAIL defaults to "" — tool will handle the missing recipient
    intent = detect_intent("send me the report")
    assert intent is not None
    assert intent.tool_name == "send_email"


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


# ---------------------------------------------------------------------------
# Negation-aware intent detection (#74)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "don't search for that",
    "do not search for anything",
    "no I don't need a search",
    "skip the search",
    "stop searching",
    "never search for that",
])
def test_negation_suppresses_search(text):
    intent = detect_intent(text)
    assert intent is None or intent.tool_name != "web_search"


@pytest.mark.parametrize("text", [
    "don't check the weather",
    "I don't need the weather",
    "no weather please",
    "skip the forecast",
])
def test_negation_suppresses_weather(text):
    intent = detect_intent(text)
    assert intent is None or intent.tool_name != "get_weather"


@pytest.mark.parametrize("text", [
    "don't email that",
    "do not send the email",
    "no email",
    "skip the mail",
])
def test_negation_suppresses_email(text):
    intent = detect_intent(text)
    assert intent is None or intent.tool_name != "send_email"


def test_negation_does_not_affect_positive():
    """Positive requests should still work fine."""
    intent = detect_intent("search for pizza recipes")
    assert intent is not None
    assert intent.tool_name == "web_search"


# ---------------------------------------------------------------------------
# Deep search detection (#81)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expect_deep", [
    ("research quantum computing", True),
    ("deep search for AI safety papers", True),
    ("do a thorough search on climate change", True),
    ("give me a comprehensive search on Python 3.12", True),
    ("search for pizza recipes", False),
    ("find me a recipe for tacos", False),
    ("google best restaurants nearby", False),
])
def test_deep_search_detection(text, expect_deep):
    """Deep search keywords should set deep=True in args."""
    intent = detect_intent(text)
    assert intent is not None
    assert intent.tool_name == "web_search"
    assert intent.args.get("deep", False) == expect_deep


# ---------------------------------------------------------------------------
# Follow-up detection (#79)
# ---------------------------------------------------------------------------

def test_followup_another_one():
    """'another one' after an image request should repeat."""
    import vox.tools as t
    # Simulate a prior image intent
    t._last_tool = "generate_image"
    t._last_tool_args = {"prompt": "sunset", "_selfie": False}
    intent = detect_intent("another one")
    assert intent is not None
    assert intent.tool_name == "generate_image"
    assert intent.args["prompt"] == "sunset"
    # Clean up
    t._last_tool = None
    t._last_tool_args = {}


def test_followup_again():
    """'again' should repeat last tool."""
    import vox.tools as t
    t._last_tool = "web_search"
    t._last_tool_args = {"query": "RTX 4090 specs"}
    intent = detect_intent("again")
    assert intent is not None
    assert intent.tool_name == "web_search"
    t._last_tool = None
    t._last_tool_args = {}


def test_followup_no_last_tool():
    """'another one' with no previous tool should not match."""
    import vox.tools as t
    t._last_tool = None
    intent = detect_intent("another one")
    assert intent is None


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
        "Unexpected web_fetch intent for generic download request"
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


# ---------------------------------------------------------------------------
# Selfie / persona detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "send me a selfie",
    "take a selfie",
    "take a pic",
    "take a picture at the beach",
    "take a photo",
    "show me a picture of yourself",
    "what do you look like?",
    "show me yourself",
    "let me see you",
    "send me a pic of yourself",
    "send me a photo of you",
])
def test_selfie_detection_matches(text):
    assert _is_selfie_request(text), f"Expected selfie detection for: {text}"


@pytest.mark.parametrize("text", [
    "generate an image of a sunset",
    "draw me a landscape",
    "take the garbage out",
    "send me the report",
    "show me the weather",
    "what does Python look like?",
])
def test_selfie_detection_no_false_positives(text):
    assert not _is_selfie_request(text), f"Unexpected selfie detection for: {text}"


@pytest.mark.parametrize("text", [
    "send me a selfie",
    "take a pic",
    "take a picture at the beach",
    "what do you look like?",
    "show me yourself",
    "let me see you",
    "send me a pic of yourself",
])
def test_selfie_intent_detected_as_generate_image(text):
    intent = detect_intent(text)
    assert intent is not None, f"Expected intent for: {text}"
    assert intent.tool_name == "generate_image"


@pytest.mark.parametrize("text", [
    "send me a selfie",
    "take a pic",
    "what do you look like?",
    "take a selfie at the beach",
])
def test_selfie_validator_allows(text):
    assert validate_tool_call("generate_image", text) is True


def test_persona_prompt_building(monkeypatch):
    """Persona prompt should include description + scene context + style."""
    monkeypatch.setattr("vox.config.VOX_PERSONA_DESCRIPTION", "a woman with brown hair and green eyes")
    monkeypatch.setattr("vox.config.VOX_PERSONA_STYLE", "photorealistic, 8k")
    prompt = _build_persona_prompt("take a selfie at the beach")
    assert "brown hair" in prompt
    assert "beach" in prompt
    assert "8k" in prompt


def test_persona_prompt_strips_conversational_fluff(monkeypatch):
    """Conversational text should be stripped, leaving only scene modifiers."""
    monkeypatch.setattr("vox.config.VOX_PERSONA_DESCRIPTION", "a woman with brown hair")
    monkeypatch.setattr("vox.config.VOX_PERSONA_STYLE", "photorealistic")
    prompt = _build_persona_prompt("I like the way you look, can I have a full body image of yourself?")
    assert "I like" not in prompt
    assert "brown hair" in prompt
    assert "full body" in prompt
    assert "photorealistic" in prompt


def test_persona_prompt_no_description(monkeypatch):
    """Without persona description, prompt should still work."""
    monkeypatch.setattr("vox.config.VOX_PERSONA_DESCRIPTION", "")
    monkeypatch.setattr("vox.config.VOX_PERSONA_STYLE", "photorealistic")
    prompt = _build_persona_prompt("take a selfie")
    assert "photorealistic" in prompt


def test_selfie_and_email_chaining(monkeypatch):
    """'send me a selfie and email it to addr' should detect both intents.

    Note: email chaining with selfie is suppressed unless an explicit
    email address appears in the message (fix #87).
    """
    monkeypatch.setattr("vox.config.USER_EMAIL", "user@example.com")
    text = "send me a selfie and email it to user@example.com"
    intents = detect_all_intents(text)
    tool_names = [i.tool_name for i in intents]
    assert "generate_image" in tool_names
    assert "send_email" in tool_names


# ---------------------------------------------------------------------------
# Dual-model routing (SFW → SDXL, NSFW → Juggernaut)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,selfie,unlocked,expected", [
    # Selfie + unlocked routes to NSFW model
    ("a woman at the beach, photorealistic", True, True, True),
    # NSFW keywords + unlocked route to NSFW model
    ("a naked woman on a bed", False, True, True),
    ("a nude portrait, artistic", False, True, True),
    ("a sexy woman in lingerie", False, True, True),
    # SFW content stays on SDXL base even when unlocked
    ("a sunset over the ocean", False, True, False),
    ("a cat sitting on a windowsill", False, False, False),
    ("a futuristic city at night", False, False, False),
    ("a portrait of a woman in a dress", False, False, False),
    # Without easter egg unlock — NEVER NSFW
    ("a naked woman on a bed", False, False, False),
    ("a woman at the beach, photorealistic", True, False, False),
])
def test_nsfw_model_routing(prompt, selfie, unlocked, expected, monkeypatch):
    monkeypatch.setattr("vox.config.IMAGE_NSFW_FILTER", "off")
    assert _should_use_nsfw_model(prompt, selfie, unlocked) is expected


def test_nsfw_routing_disabled_when_filter_on(monkeypatch):
    """When NSFW filter is ON, never route to NSFW model."""
    monkeypatch.setattr("vox.config.IMAGE_NSFW_FILTER", "on")
    assert _should_use_nsfw_model("a naked woman", False) is False
    assert _should_use_nsfw_model("selfie prompt", True) is False


# ---------------------------------------------------------------------------
# Implicit NSFW image intent — no "image/picture" keyword needed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "show me a naked woman",
    "give me some sexy girls in bikinis",
    "make a nude portrait",
    "draw me a topless brunette",
    "create a seductive pose",
    "get me a sexy blonde",
    "show me a hot girl in lingerie",
])
def test_nsfw_implicit_image_intent(text):
    """Mature content keywords + action verb should trigger image generation."""
    intent = detect_intent(text)
    assert intent is not None, f"Expected image intent for: {text}"
    assert intent.tool_name == "generate_image"


@pytest.mark.parametrize("text", [
    "tell me about lingerie brands",
    "what is the definition of nude in art?",
    "how do bikinis affect body image?",
])
def test_nsfw_no_false_positive_without_verb(text):
    """Mature words without visual verb should NOT trigger image generation."""
    intent = detect_intent(text)
    if intent is not None:
        assert intent.tool_name != "generate_image", (
            f"Unexpected image intent for informational query: {text}"
        )


def test_nsfw_prompt_extraction():
    """Mature image request should produce a clean prompt."""
    prompt = _extract_image_prompt("make a nude portrait of a woman")
    assert "nude" in prompt.lower()
    assert "make a" not in prompt.lower()


def test_nsfw_keywords_route_to_nsfw_model(monkeypatch):
    """Content keywords + easter egg unlock should route to unrestricted model."""
    monkeypatch.setattr("vox.config.IMAGE_NSFW_FILTER", "off")
    # With unlock → NSFW routes
    assert _should_use_nsfw_model("nude portrait", False, nsfw_unlocked=True) is True
    assert _should_use_nsfw_model("a sexy woman in lingerie", False, nsfw_unlocked=True) is True
    assert _should_use_nsfw_model("topless at the beach", False, nsfw_unlocked=True) is True
    # Without unlock → always SFW
    assert _should_use_nsfw_model("nude portrait", False, nsfw_unlocked=False) is False
    assert _should_use_nsfw_model("topless at the beach", True, nsfw_unlocked=False) is False


# ---------------------------------------------------------------------------
# System commands (#42)
# ---------------------------------------------------------------------------

def test_run_command_disk_space():
    result = execute_tool("run_command", {"command": "disk_space"})
    # Should return disk info (drive letters or df output)
    assert len(result) > 10


def test_run_command_unknown():
    result = execute_tool("run_command", {"command": "rm -rf /"})
    assert "Unknown command" in result or "Available" in result


def test_run_command_empty():
    result = execute_tool("run_command", {"command": ""})
    assert "Available" in result


def test_disk_space_intent():
    intent = detect_intent("how much disk space do I have")
    assert intent is not None
    assert intent.tool_name == "run_command"
    assert intent.args["command"] == "disk_space"


def test_gpu_processes_intent():
    intent = detect_intent("run nvidia-smi")
    assert intent is not None
    assert intent.tool_name == "run_command"
    assert intent.args["command"] == "gpu_processes"


# ---------------------------------------------------------------------------
# Notes / To-Do (#43)
# ---------------------------------------------------------------------------

def test_add_note(tmp_path, monkeypatch):
    """add_note should save a note to the JSON store."""
    monkeypatch.setattr("vox.tools._NOTES_FILE", tmp_path / "notes.json")
    result = execute_tool("add_note", {"text": "buy oil filter"})
    assert "saved" in result.lower() or "got it" in result.lower()
    assert "buy oil filter" in result


def test_list_notes(tmp_path, monkeypatch):
    """list_notes should return saved notes."""
    monkeypatch.setattr("vox.tools._NOTES_FILE", tmp_path / "notes.json")
    execute_tool("add_note", {"text": "buy oil filter"})
    execute_tool("add_note", {"text": "pick up parts"})
    result = execute_tool("list_notes", {})
    assert "buy oil filter" in result
    assert "pick up parts" in result


def test_complete_note(tmp_path, monkeypatch):
    """complete_note should mark a note as done."""
    monkeypatch.setattr("vox.tools._NOTES_FILE", tmp_path / "notes.json")
    execute_tool("add_note", {"text": "test note"})
    result = execute_tool("complete_note", {"note_id": 1})
    assert "done" in result.lower() or "complete" in result.lower()


def test_note_intent_detection():
    """Note-related phrases should trigger add_note."""
    intent = detect_intent("take a note: buy milk")
    assert intent is not None
    assert intent.tool_name == "add_note"
    assert "buy milk" in intent.args.get("text", "")


def test_list_notes_intent():
    intent = detect_intent("what are my notes")
    assert intent is not None
    assert intent.tool_name == "list_notes"


# ---------------------------------------------------------------------------
# Screenshot intent detection (#41)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "take a screenshot",
    "screenshot please",
    "capture my screen",
    "what's on my screen",
])
def test_screenshot_intent(text):
    intent = detect_intent(text)
    assert intent is not None
    assert intent.tool_name == "take_screenshot"


# ---------------------------------------------------------------------------
# Batch image count extraction (#57)
# ---------------------------------------------------------------------------

def test_extract_image_count_basic():
    from vox.tools import _extract_image_count
    assert _extract_image_count("give me 5 pictures of cats") == 5
    assert _extract_image_count("generate 3 images of dogs") == 3
    assert _extract_image_count("take a selfie") == 1
    assert _extract_image_count("draw me a picture of the moon") == 1


def test_extract_image_count_capped():
    from vox.tools import _extract_image_count
    assert _extract_image_count("make 50 images of cats") == 10


def test_batch_intent_detection():
    """Count-based image requests should include count in args."""
    intent = detect_intent("generate 5 images of a sunset over the ocean")
    assert intent is not None
    assert intent.tool_name == "generate_image"
    assert intent.args.get("count", 1) == 5


# ---------------------------------------------------------------------------
# Image progress callback
# ---------------------------------------------------------------------------

def test_image_progress_callback_settable():
    """The progress callback global should be settable and clearable."""
    from vox import tools as tools_mod
    assert tools_mod._image_progress_fn is None
    calls = []
    tools_mod._image_progress_fn = lambda step, total: calls.append((step, total))
    tools_mod._image_progress_fn(3, 30)
    assert calls == [(3, 30)]
    tools_mod._image_progress_fn = None
    assert tools_mod._image_progress_fn is None


# ---------------------------------------------------------------------------
# Contacts intent detection (#47)
# ---------------------------------------------------------------------------

def test_add_contact_intent():
    intent = detect_intent("add contact: Mike, mechanic, 555-1234")
    assert intent is not None
    assert intent.tool_name == "add_contact"


def test_lookup_contact_intent():
    intent = detect_intent("what's John's phone number?")
    assert intent is not None
    assert intent.tool_name == "lookup_contact"


def test_lookup_contact_email():
    intent = detect_intent("what is Sarah's email address?")
    assert intent is not None
    assert intent.tool_name == "lookup_contact"


def test_list_contacts_intent():
    intent = detect_intent("show my contacts")
    assert intent is not None
    assert intent.tool_name == "list_contacts"


def test_remove_contact_intent():
    intent = detect_intent("remove contact #3")
    assert intent is not None
    assert intent.tool_name == "remove_contact"
    assert intent.args.get("contact_id") == 3


def test_contact_tool_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.contacts._CONTACTS_FILE", tmp_path / "contacts.json")
    result = execute_tool("add_contact", {"name": "Test User", "email": "test@example.com"})
    assert "saved" in result.lower()
    result = execute_tool("lookup_contact", {"query": "Test"})
    assert "Test User" in result
    result = execute_tool("list_contacts", {})
    assert "Test User" in result
    result = execute_tool("remove_contact", {"contact_id": 1})
    assert "removed" in result.lower()


# ---------------------------------------------------------------------------
# Clipboard intent detection (#68)
# ---------------------------------------------------------------------------

def test_read_clipboard_intent():
    intent = detect_intent("what's on my clipboard?")
    assert intent is not None
    assert intent.tool_name == "read_clipboard"


def test_read_clipboard_intent_alt():
    intent = detect_intent("read my clipboard")
    assert intent is not None
    assert intent.tool_name == "read_clipboard"


def test_write_clipboard_intent():
    intent = detect_intent("copy that to clipboard")
    assert intent is not None
    assert intent.tool_name == "write_clipboard"


def test_clipboard_execution():
    result = execute_tool("read_clipboard", {})
    # Should return something (either content or "empty")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Response caching (#36)
# ---------------------------------------------------------------------------

def test_cache_hit():
    from vox.tools import _cache_set, cache_bust
    cache_bust()  # clear
    # Manually populate cache
    _cache_set("get_weather", {"location": "NYC"}, "Sunny and 72F")
    # execute_tool should return cached result
    result = execute_tool("get_weather", {"location": "NYC"})
    assert result == "Sunny and 72F"
    cache_bust()


def test_cache_bust():
    from vox.tools import _cache_set, cache_bust
    cache_bust()
    _cache_set("get_weather", {"location": "NYC"}, "Sunny")
    cache_bust("get_weather")
    # After bust, cache should be empty for that tool
    from vox.tools import _cache_get
    assert _cache_get("get_weather", {"location": "NYC"}) is None


def test_cache_bust_all():
    from vox.tools import _cache_get, _cache_set, cache_bust
    cache_bust()
    _cache_set("get_weather", {"location": "NYC"}, "Sunny")
    _cache_set("get_system_info", {}, "RTX 3090")
    cache_bust()
    assert _cache_get("get_weather", {"location": "NYC"}) is None
    assert _cache_get("get_system_info", {}) is None


def test_refresh_busts_cache():
    """'refresh' keyword should clear cache before detecting intent."""
    from vox.tools import _cache_get, _cache_set, cache_bust
    cache_bust()
    _cache_set("get_weather", {"location": "NYC"}, "Old weather")
    # Detecting intent with "refresh" should clear cache
    detect_intent("refresh the weather")
    assert _cache_get("get_weather", {"location": "NYC"}) is None


def test_non_cached_tools_skip_cache():
    from vox.tools import _cache_get, cache_bust
    cache_bust()
    # generate_image is not in _TOOL_TTL, so should not be cached
    assert _cache_get("generate_image", {"prompt": "test"}) is None


# ---------------------------------------------------------------------------
# Code runner (#70)
# ---------------------------------------------------------------------------

def test_calculate_intent():
    intent = detect_intent("calculate 15 plus 27")
    assert intent is not None
    assert intent.tool_name == "run_code"


def test_convert_intent():
    intent = detect_intent("convert 72F to C")
    assert intent is not None
    assert intent.tool_name == "run_code"


def test_run_python_intent():
    intent = detect_intent("run python print('hello')")
    assert intent is not None
    assert intent.tool_name == "run_code"


def test_run_code_math():
    result = execute_tool("run_code", {"expression": "2 + 2"})
    assert "4" in result


def test_run_code_temp_conversion():
    result = execute_tool("run_code", {"expression": "72F to C"})
    assert "22" in result  # 72F = 22.2C


def test_run_code_python():
    result = execute_tool("run_code", {"code": "print(sum(range(10)))"})
    assert "45" in result


def test_run_code_timeout():
    result = execute_tool("run_code", {"code": "import time; time.sleep(30)"})
    assert "timed out" in result.lower()


def test_run_code_empty():
    result = execute_tool("run_code", {})
    assert "no code" in result.lower()


# ---------------------------------------------------------------------------
# Image upscaling (#64)
# ---------------------------------------------------------------------------

def test_upscale_intent():
    intent = detect_intent("upscale this image photo.png")
    assert intent is not None
    assert intent.tool_name == "upscale_image"


def test_upscale_alt_intent():
    intent = detect_intent("make this picture bigger")
    assert intent is not None
    assert intent.tool_name == "upscale_image"


def test_upscale_no_file():
    result = execute_tool("upscale_image", {})
    assert "no image" in result.lower()


def test_upscale_missing_file():
    result = execute_tool("upscale_image", {"path": "nonexistent_file.png"})
    assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# File navigator (#32)
# ---------------------------------------------------------------------------

def test_list_files_intent():
    intent = detect_intent("what's in my downloads?")
    assert intent is not None
    assert intent.tool_name == "list_files"
    assert intent.args.get("directory") == "downloads"


def test_find_file_intent():
    intent = detect_intent("find that report.pdf")
    assert intent is not None
    assert intent.tool_name == "find_file"


def test_latest_download_intent():
    intent = detect_intent("latest download")
    assert intent is not None
    assert intent.tool_name == "list_files"
    assert intent.args.get("sort") == "newest"


def test_list_files_execution():
    # Should return something even if folder is empty
    result = execute_tool("list_files", {"directory": "downloads"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_find_file_empty():
    result = execute_tool("find_file", {})
    assert "what file" in result.lower()


# ---------------------------------------------------------------------------
# Daily briefing (#29)
# ---------------------------------------------------------------------------

def test_briefing_intent():
    intent = detect_intent("good morning")
    assert intent is not None
    assert intent.tool_name == "daily_briefing"


def test_briefing_execution():
    result = execute_tool("daily_briefing", {})
    assert isinstance(result, str)
    # Should at least have the time
    assert len(result) > 10


def test_upscale_with_pil(tmp_path):
    """Test actual upscaling with PIL Lanczos fallback."""
    pytest.importorskip("PIL", reason="Pillow not installed")
    from PIL import Image
    # Create a small test image
    img = Image.new("RGB", (64, 64), color="red")
    img_path = tmp_path / "test_small.png"
    img.save(str(img_path))
    # Monkeypatch DOWNLOADS_DIR to tmp_path
    import vox.config
    orig = vox.config.DOWNLOADS_DIR
    vox.config.DOWNLOADS_DIR = tmp_path
    try:
        result = execute_tool("upscale_image", {"path": str(img_path), "scale": 2})
        assert "upscaled" in result.lower()
        assert "128x128" in result
    finally:
        vox.config.DOWNLOADS_DIR = orig


# ---------------------------------------------------------------------------
# Macros intent detection (#34)
# ---------------------------------------------------------------------------

def test_run_macro_intent():
    intent = detect_intent("run my macro morning briefing")
    assert intent is not None
    assert intent.tool_name == "run_macro"
    assert "morning briefing" in intent.args.get("name", "")


def test_create_macro_intent():
    intent = detect_intent("create a macro called morning briefing")
    assert intent is not None
    assert intent.tool_name == "add_macro"


def test_list_macros_intent():
    intent = detect_intent("show my macros")
    assert intent is not None
    assert intent.tool_name == "list_macros"


def test_macro_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.macros._MACROS_FILE", tmp_path / "macros.json")
    result = execute_tool("add_macro", {
        "_raw": "create macro time check: get_current_time",
    })
    assert "saved" in result.lower()
    result = execute_tool("run_macro", {"name": "time check"})
    assert "time check" in result.lower()
    result = execute_tool("list_macros", {})
    assert "time check" in result.lower()


# ---------------------------------------------------------------------------
# Reminder intent detection + execution
# ---------------------------------------------------------------------------

def test_set_reminder_intent():
    intent = detect_intent("remind me in 30 minutes to check the oven")
    assert intent is not None
    assert intent.tool_name == "set_reminder"


def test_set_reminder_intent_timer():
    intent = detect_intent("set a timer for 10 minutes")
    assert intent is not None
    assert intent.tool_name == "set_reminder"


def test_set_reminder_intent_alarm():
    intent = detect_intent("set an alarm for 2 hours")
    assert intent is not None
    assert intent.tool_name == "set_reminder"


def test_list_reminders_intent():
    intent = detect_intent("show my reminders")
    assert intent is not None
    assert intent.tool_name == "list_reminders"


def test_list_reminders_intent_timers():
    intent = detect_intent("what are my timers")
    assert intent is not None
    assert intent.tool_name == "list_reminders"


def test_set_reminder_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.tools._get_reminders_file", lambda: tmp_path / "reminders.json")
    result = execute_tool("set_reminder", {"_raw": "remind me in 30 minutes to check the oven"})
    assert "30 minute" in result
    assert "check the oven" in result


def test_set_reminder_default_time(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.tools._get_reminders_file", lambda: tmp_path / "reminders.json")
    result = execute_tool("set_reminder", {"text": "take out the trash"})
    assert "30 minute" in result  # default


def test_list_reminders_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.tools._get_reminders_file", lambda: tmp_path / "reminders.json")
    result = execute_tool("list_reminders", {})
    assert "no pending" in result.lower()


def test_list_reminders_with_items(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.tools._get_reminders_file", lambda: tmp_path / "reminders.json")
    execute_tool("set_reminder", {"text": "feed the cat", "minutes": 60})
    result = execute_tool("list_reminders", {})
    assert "feed the cat" in result
    assert "1 reminder" in result


def test_check_reminders_fires(tmp_path, monkeypatch):
    import datetime
    monkeypatch.setattr("vox.tools._get_reminders_file", lambda: tmp_path / "reminders.json")
    # Set reminder due in the past
    execute_tool("set_reminder", {"text": "past event", "minutes": 1})
    # Manually backdate it
    import json
    reminders = json.loads((tmp_path / "reminders.json").read_text())
    reminders[0]["due_at"] = (datetime.datetime.now() - datetime.timedelta(minutes=5)).isoformat()
    (tmp_path / "reminders.json").write_text(json.dumps(reminders))
    from vox.tools import check_reminders
    fired = check_reminders()
    assert len(fired) == 1
    assert "past event" in fired[0]
    # Should be removed after firing
    fired2 = check_reminders()
    assert len(fired2) == 0


def test_parse_time_offset():
    from vox.tools import _parse_time_offset
    assert _parse_time_offset("in 30 minutes") == 30
    assert _parse_time_offset("in 2 hours") == 120
    assert _parse_time_offset("5 min") == 5
    assert _parse_time_offset("no time here") == 0


def test_validate_set_reminder():
    assert validate_tool_call("set_reminder", "remind me to call mom") is True
    assert validate_tool_call("set_reminder", "what's the weather") is False


def test_validate_list_reminders():
    assert validate_tool_call("list_reminders", "show my reminders") is True
    assert validate_tool_call("list_reminders", "what's the weather") is False


# ---------------------------------------------------------------------------
# News / RSS
# ---------------------------------------------------------------------------

def test_news_intent():
    intent = detect_intent("what's in the news today")
    assert intent is not None
    assert intent.tool_name == "get_news"


def test_news_intent_tech():
    intent = detect_intent("show me the tech headlines")
    assert intent is not None
    assert intent.tool_name == "get_news"


def test_news_intent_negative():
    intent = detect_intent("what's the weather")
    assert intent is None or intent.tool_name != "get_news"


def test_validate_get_news():
    assert validate_tool_call("get_news", "what's in the news") is True
    assert validate_tool_call("get_news", "what's the weather") is False


def test_parse_rss():
    from vox.tools import _parse_rss
    sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Breaking: Test Headline</title>
          <link>https://example.com/article1</link>
          <description>This is a test article about something important.</description>
        </item>
        <item>
          <title>Another Story</title>
          <link>https://example.com/article2</link>
          <description>Second article &lt;b&gt;with HTML&lt;/b&gt; tags.</description>
        </item>
      </channel>
    </rss>"""
    items = _parse_rss(sample_rss)
    assert len(items) == 2
    assert items[0]["title"] == "Breaking: Test Headline"
    assert items[1]["link"] == "https://example.com/article2"
    assert "<b>" not in items[1]["description"]


def test_parse_rss_atom():
    from vox.tools import _parse_rss
    sample_atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Atom Feed</title>
      <entry>
        <title>Atom Article</title>
        <link href="https://example.com/atom1"/>
        <summary>Summary of atom article.</summary>
      </entry>
    </feed>"""
    items = _parse_rss(sample_atom)
    assert len(items) == 1
    assert items[0]["title"] == "Atom Article"
    assert items[0]["link"] == "https://example.com/atom1"


def test_parse_rss_empty():
    from vox.tools import _parse_rss
    assert _parse_rss("not xml at all") == []
    assert _parse_rss("<root></root>") == []


def test_parse_rss_max_items():
    from vox.tools import _parse_rss
    items_xml = "".join(
        f'<item><title>Item {i}</title><link/><description/></item>'
        for i in range(20)
    )
    rss = f'<rss><channel>{items_xml}</channel></rss>'
    items = _parse_rss(rss, max_items=3)
    assert len(items) == 3


# ---------------------------------------------------------------------------
# History condensation (issue #13 — context leak prevention)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Media control
# ---------------------------------------------------------------------------

def test_media_play_intent():
    intent = detect_intent("play some music")
    assert intent is not None
    assert intent.tool_name == "media_control"


def test_media_pause_intent():
    intent = detect_intent("pause the music")
    assert intent is not None
    assert intent.tool_name == "media_control"


def test_media_skip_intent():
    intent = detect_intent("skip this song")
    assert intent is not None
    assert intent.tool_name == "media_control"


def test_media_now_playing_intent():
    intent = detect_intent("what's playing")
    assert intent is not None
    assert intent.tool_name == "media_control"


def test_media_volume_intent():
    intent = detect_intent("volume up")
    assert intent is not None
    assert intent.tool_name == "media_control"


def test_validate_media_control():
    assert validate_tool_call("media_control", "play some music") is True
    assert validate_tool_call("media_control", "what's the weather") is False


# ---------------------------------------------------------------------------
# Smart home / Home Assistant
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Document search (RAG)
# ---------------------------------------------------------------------------

def test_search_documents_intent():
    intent = detect_intent("what does my lease say about pets")
    assert intent is not None
    assert intent.tool_name == "search_documents"


def test_index_documents_intent():
    intent = detect_intent("index my documents")
    assert intent is not None
    assert intent.tool_name == "index_documents"


def test_validate_search_documents():
    assert validate_tool_call("search_documents", "search my lease for pet policy") is True
    assert validate_tool_call("search_documents", "what's the weather") is False


def test_search_documents_no_deps():
    """Should gracefully handle missing chromadb."""
    result = execute_tool("search_documents", {"query": "test query"})
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Smart home / Home Assistant
# ---------------------------------------------------------------------------

def test_smart_home_turn_on_intent():
    intent = detect_intent("turn on the living room lights")
    assert intent is not None
    assert intent.tool_name == "smart_home"


def test_smart_home_thermostat_intent():
    intent = detect_intent("set thermostat to 72")
    assert intent is not None
    assert intent.tool_name == "smart_home"


def test_smart_home_lock_intent():
    intent = detect_intent("lock the front door")
    assert intent is not None
    assert intent.tool_name == "smart_home"


def test_smart_home_status_intent():
    intent = detect_intent("is the garage door open?")
    assert intent is not None
    assert intent.tool_name == "smart_home"


def test_validate_smart_home():
    assert validate_tool_call("smart_home", "turn off the bedroom light") is True
    assert validate_tool_call("smart_home", "what's the weather") is False


def test_smart_home_not_configured():
    result = execute_tool("smart_home", {"action": "toggle", "entity_hint": "light", "state": "on"})
    assert "not configured" in result.lower()


def test_guess_domain():
    from vox.tools import _guess_domain
    assert _guess_domain("living room light") == "light"
    assert _guess_domain("ceiling fan") == "fan"
    assert _guess_domain("smart plug") == "switch"
    assert _guess_domain("window blinds") == "cover"


def test_parse_volume_action():
    from vox.tools import _parse_volume_action
    assert _parse_volume_action("volume up") == "volume_up"
    assert _parse_volume_action("volume down") == "volume_down"
    assert _parse_volume_action("mute") == "mute"
    assert _parse_volume_action("unmute") == "unmute"
    assert _parse_volume_action("louder") == "volume_up"
    assert _parse_volume_action("softer") == "volume_down"


def test_media_now_playing_execution():
    result = execute_tool("media_control", {"action": "now_playing"})
    assert isinstance(result, str)


def test_media_unknown_action():
    result = execute_tool("media_control", {"action": "bogus"})
    assert "unknown" in result.lower() or "Unknown" in result


# ---------------------------------------------------------------------------
# Search backend pluggable architecture
# ---------------------------------------------------------------------------

def test_search_ddg_returns_tuple():
    from vox.tools import _search_ddg
    # Just verify the function signature works (actual network call may fail)
    try:
        results, urls = _search_ddg("test query")
        assert isinstance(results, list)
        assert isinstance(urls, list)
    except Exception:  # noqa: S110
        pass  # Network may not be available in CI


def test_search_brave_no_key():
    from vox.tools import _search_brave
    # Without API key, should return empty
    results, urls = _search_brave("test")
    assert results == []
    assert urls == []


def test_search_searxng_no_url():
    from vox.tools import _search_searxng
    # Without URL configured, should return empty
    results, urls = _search_searxng("test")
    assert results == []
    assert urls == []


def test_search_fallback(monkeypatch):
    """If primary engine fails, falls back to DDG."""
    monkeypatch.setattr("vox.config.SEARCH_ENGINE", "brave")
    monkeypatch.setattr("vox.config.BRAVE_API_KEY", "fake-key")
    # Brave will fail (bad key), should fall back to DDG
    result = execute_tool("web_search", {"query": "test"})
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# History condensation (issue #13 — context leak prevention)
# ---------------------------------------------------------------------------

def test_condense_old_tool_results():
    from vox.llm import _condense_old_tool_results, _history
    _history.clear()
    # Simulate a conversation with tool results
    _history.extend([
        {"role": "user", "content": "what's the weather"},
        {"role": "tool", "content": "Weather in Sherman TX: 72°F, partly cloudy, wind 5mph NW, humidity 45%, " * 3},
        {"role": "assistant", "content": "It's 72°F and partly cloudy!"},
        {"role": "user", "content": "show me a selfie"},
        {"role": "tool", "content": "Image generated: /path/to/selfie.png with prompt 'beautiful woman' " * 3},
        {"role": "assistant", "content": "Here you go!"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hey!"},
    ])
    _condense_old_tool_results()
    # Old tool results (indices 1 and 4) should be condensed
    assert _history[1]["content"].startswith("[previous result:")
    assert len(_history[1]["content"]) < 120
    # Recent messages (last 4) should be unchanged
    assert _history[-1]["content"] == "Hey!"
    assert _history[-2]["content"] == "hello"
    _history.clear()


def test_condense_preserves_recent_tools():
    from vox.llm import _condense_old_tool_results, _history
    _history.clear()
    _history.extend([
        {"role": "user", "content": "old message"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "what time is it"},
        {"role": "tool", "content": "Current time: 2:30 PM, March 5 2026"},
        {"role": "assistant", "content": "It's 2:30 PM!"},
    ])
    _condense_old_tool_results()
    # The tool result at index 3 is within last 4 messages, should be preserved
    assert _history[3]["content"] == "Current time: 2:30 PM, March 5 2026"
    _history.clear()

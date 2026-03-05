"""Tests for tool execution, intent detection, and validation."""

import pytest

from vox.tools import detect_intent, execute_tool, validate_tool_call


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


def test_send_email_no_config():
    """Email tool should fail gracefully when SMTP is not configured."""
    result = execute_tool("send_email", {"to": "test@example.com", "subject": "Test", "body": "Hi"})
    assert "not configured" in result or "Failed" in result


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
    ("email me at rayketcham@ogjos.com", "send_email"),
    ("can you email the results to test@example.com", "send_email"),
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
])
def test_validate_tool_call_blocks_irrelevant(tool, text):
    assert validate_tool_call(tool, text) is False


def test_validate_unknown_tool_passes_through():
    """Unknown tools should pass validation (execute_tool handles the error)."""
    assert validate_tool_call("nonexistent_tool", "anything") is True

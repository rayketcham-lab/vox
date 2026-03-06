"""Tests for persona system — card loading, prompt building, moods, appearance."""

from unittest.mock import patch

from vox import persona

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_card(**overrides):
    """Build a minimal persona card dict."""
    card = {
        "name": "TestBot",
        "identity": {
            "personality": "Friendly and helpful.",
            "speech_style": "Casual and warm.",
        },
        "backstory": "A test persona.",
        "rules": {},
    }
    card.update(overrides)
    return card


# ---------------------------------------------------------------------------
# Card loading
# ---------------------------------------------------------------------------

def test_load_card_valid(tmp_path):
    card_file = tmp_path / "test.yaml"
    card_file.write_text("name: TestBot\nidentity:\n  personality: Friendly\n")
    card = persona.load_card(card_file)
    assert card["name"] == "TestBot"
    assert persona.get_card() is not None


def test_load_card_missing(tmp_path):
    card = persona.load_card(tmp_path / "nonexistent.yaml")
    assert card == {}


def test_get_card_default():
    """Before any card is loaded, get_card may return None or previous card."""
    result = persona.get_card()
    # Just verify it returns without error
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Time periods
# ---------------------------------------------------------------------------

@patch("vox.persona.datetime")
def test_get_time_period_morning(mock_dt):
    mock_dt.now.return_value.hour = 8
    assert persona._get_time_period() == "morning"


@patch("vox.persona.datetime")
def test_get_time_period_afternoon(mock_dt):
    mock_dt.now.return_value.hour = 14
    assert persona._get_time_period() == "afternoon"


@patch("vox.persona.datetime")
def test_get_time_period_evening(mock_dt):
    mock_dt.now.return_value.hour = 19
    assert persona._get_time_period() == "evening"


@patch("vox.persona.datetime")
def test_get_time_period_night(mock_dt):
    mock_dt.now.return_value.hour = 2
    assert persona._get_time_period() == "night"


@patch("vox.persona.datetime")
def test_get_time_period_custom_schedule(mock_dt):
    mock_dt.now.return_value.hour = 9
    card = {"schedule": {"hours": {"morning": [8, 10]}}}
    assert persona._get_time_period(card) == "morning"


@patch("vox.persona.datetime")
def test_get_time_period_boundary(mock_dt):
    """Hour 5 should be start of morning (default range [5, 11])."""
    mock_dt.now.return_value.hour = 5
    assert persona._get_time_period() == "morning"


@patch("vox.persona.datetime")
def test_get_time_period_boundary_exclusive(mock_dt):
    """Hour 11 should NOT be morning (range is [5, 11) exclusive upper)."""
    mock_dt.now.return_value.hour = 11
    assert persona._get_time_period() == "afternoon"


# ---------------------------------------------------------------------------
# Mood block
# ---------------------------------------------------------------------------

@patch("vox.persona.datetime")
def test_get_mood_block_with_mood(mock_dt):
    mock_dt.now.return_value.hour = 8
    mock_dt.now.return_value.day = 1
    card = {
        "moods": {
            "morning": {"vibe": "cheerful", "energy": "high"},
        },
    }
    result = persona._get_mood_block(card)
    assert "morning" in result
    assert "cheerful" in result
    assert "high" in result


@patch("vox.persona.datetime")
def test_get_mood_block_no_mood(mock_dt):
    mock_dt.now.return_value.hour = 8
    card = {"moods": {}}
    result = persona._get_mood_block(card)
    assert result == ""


@patch("vox.persona.datetime")
def test_get_mood_block_with_activity(mock_dt):
    mock_dt.now.return_value.hour = 8
    mock_dt.now.return_value.day = 1
    card = {
        "moods": {"morning": {"vibe": "happy", "energy": "medium"}},
        "activities": {"morning": ["having coffee", "stretching"]},
    }
    result = persona._get_mood_block(card)
    assert "Right now you're" in result


# ---------------------------------------------------------------------------
# Memory block
# ---------------------------------------------------------------------------

def test_get_memory_block_empty():
    assert persona._get_memory_block({}) == ""
    assert persona._get_memory_block({"memory_fragments": []}) == ""


def test_get_memory_block_with_fragments():
    card = {"memory_fragments": ["Likes cats", "Hates rain"]}
    result = persona._get_memory_block(card)
    assert "Likes cats" in result
    assert "Hates rain" in result
    assert "Things you remember" in result


# ---------------------------------------------------------------------------
# Appearance and style
# ---------------------------------------------------------------------------

def test_get_appearance_from_card(tmp_path):
    card_file = tmp_path / "app.yaml"
    card_file.write_text("name: Test\nappearance:\n  description: 'tall with red hair'\n")
    persona.load_card(card_file)
    assert "tall with red hair" in persona.get_appearance()


def test_get_appearance_strips_newlines(tmp_path):
    card_file = tmp_path / "app2.yaml"
    card_file.write_text("name: Test\nappearance:\n  description: 'line1\\nline2'\n")
    persona.load_card(card_file)
    result = persona.get_appearance()
    assert "\n" not in result


def test_get_style_tags_from_card(tmp_path):
    card_file = tmp_path / "style.yaml"
    card_file.write_text("name: Test\nappearance:\n  style_tags: 'photorealistic, cinematic'\n")
    persona.load_card(card_file)
    assert "photorealistic" in persona.get_style_tags()


# ---------------------------------------------------------------------------
# Prompt building from card
# ---------------------------------------------------------------------------

def test_build_from_card_includes_name():
    card = _minimal_card()
    result = persona._build_from_card(card)
    assert "TestBot" in result


def test_build_from_card_includes_personality():
    card = _minimal_card()
    result = persona._build_from_card(card)
    assert "Friendly and helpful" in result


def test_build_from_card_includes_backstory():
    card = _minimal_card()
    result = persona._build_from_card(card)
    assert "A test persona" in result


def test_build_from_card_includes_likes():
    card = _minimal_card()
    card["identity"]["likes"] = ["cats", "coffee"]
    result = persona._build_from_card(card)
    assert "cats" in result
    assert "coffee" in result


def test_build_from_card_includes_dislikes():
    card = _minimal_card()
    card["identity"]["dislikes"] = ["spam"]
    result = persona._build_from_card(card)
    assert "spam" in result


def test_build_from_card_includes_quirks():
    card = _minimal_card()
    card["identity"]["quirks"] = ["talks to plants"]
    result = persona._build_from_card(card)
    assert "talks to plants" in result


def test_build_from_card_favorites():
    card = _minimal_card(favorites={"music": ["jazz", "lo-fi"], "food": "pizza"})
    result = persona._build_from_card(card)
    assert "jazz" in result
    assert "pizza" in result


def test_build_from_card_opinions():
    card = _minimal_card(opinions={"pineapple_pizza": "love it"})
    result = persona._build_from_card(card)
    assert "pineapple pizza" in result
    assert "love it" in result


def test_build_from_card_never_break_character():
    card = _minimal_card(rules={"never_break_character": True})
    result = persona._build_from_card(card)
    assert "NEVER break character" in result
    assert "TestBot" in result


def test_build_from_card_never_refuse():
    card = _minimal_card(rules={"never_refuse": True})
    result = persona._build_from_card(card)
    assert "Never refuse" in result


def test_build_from_card_no_disclaimers():
    card = _minimal_card(rules={"no_disclaimers": True})
    result = persona._build_from_card(card)
    assert "No disclaimers" in result


def test_build_from_card_tool_rules():
    card = _minimal_card()
    result = persona._build_from_card(card)
    assert "TOOL USE RULES" in result


# ---------------------------------------------------------------------------
# Legacy prompt
# ---------------------------------------------------------------------------

@patch("vox.persona.VOX_PERSONA_NAME", "LegacyBot")
@patch("vox.persona.VOX_PERSONA_DESCRIPTION", "a friendly bot")
def test_build_legacy_with_name():
    result = persona._build_legacy()
    assert "LegacyBot" in result
    assert "a friendly bot" in result


@patch("vox.persona.VOX_PERSONA_NAME", "")
def test_build_legacy_no_name():
    result = persona._build_legacy()
    assert "VOX" in result
    assert "Be concise and direct" in result


# ---------------------------------------------------------------------------
# Tool rules block
# ---------------------------------------------------------------------------

@patch("vox.persona.IMAGE_NSFW_FILTER", "off")
def test_tool_rules_nsfw_off():
    result = persona._tool_rules_block()
    assert "NSFW filter is OFF" in result


@patch("vox.persona.IMAGE_NSFW_FILTER", "on")
def test_tool_rules_nsfw_on():
    result = persona._tool_rules_block()
    assert "NSFW filter is enabled" in result


def test_tool_rules_contains_all_tools():
    result = persona._tool_rules_block()
    expected_tools = [
        "get_weather", "get_current_time", "get_system_info",
        "web_search", "send_email", "generate_image",
        "set_reminder", "list_reminders", "get_news",
        "media_control", "smart_home", "search_documents",
    ]
    for tool in expected_tools:
        assert tool in result, f"Missing tool rule for {tool}"


# ---------------------------------------------------------------------------
# Full system prompt
# ---------------------------------------------------------------------------

@patch("vox.preferences.build_preferences_block", return_value="")
@patch("vox.memory.build_memory_prompt_block", return_value="")
def test_build_system_prompt_legacy(mock_mem, mock_prefs):
    """With no card loaded, should produce a legacy prompt."""
    old_card = persona._card
    persona._card = None
    try:
        result = persona.build_system_prompt()
        assert "TOOL USE RULES" in result
    finally:
        persona._card = old_card


# Cleanup: reset persona state after tests
def test_cleanup():
    """Reset module state."""
    persona._card = None
    persona._card_path = None

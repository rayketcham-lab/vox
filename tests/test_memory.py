"""Tests for persistent user memory and reminder systems."""

import pytest

from vox.memory import (
    build_memory_prompt_block,
    detect_memory_intent,
    forget,
    recall,
    remember,
)
from vox.reminders import (
    add_reminder,
    cancel_reminder,
    detect_reminder_intent,
    list_reminders,
)

# ---------------------------------------------------------------------------
# Memory intent detection
# ---------------------------------------------------------------------------

class TestMemoryIntentDetection:
    @pytest.mark.parametrize("text,expected_action", [
        ("remember that I hate cilantro", "remember"),
        ("don't forget my mom's birthday is March 12", "remember"),
        ("keep in mind I'm allergic to penicillin", "remember"),
        ("note that my favorite color is blue", "remember"),
        ("fyi I work from home on Fridays", "remember"),
    ])
    def test_remember_detected(self, text, expected_action):
        result = detect_memory_intent(text)
        assert result is not None
        assert result["action"] == expected_action
        assert len(result["data"]) > 5

    @pytest.mark.parametrize("text", [
        "forget about cilantro",
        "stop remembering about my allergy",
    ])
    def test_forget_detected(self, text):
        result = detect_memory_intent(text)
        assert result is not None
        assert result["action"] == "forget"

    @pytest.mark.parametrize("text", [
        "what do you know about me",
        "what do you remember",
        "recall what I told you",
    ])
    def test_recall_detected(self, text):
        result = detect_memory_intent(text)
        assert result is not None
        assert result["action"] == "recall"

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "show me a selfie",
        "hello how are you",
    ])
    def test_non_memory_ignored(self, text):
        assert detect_memory_intent(text) is None


class TestMemoryStorage:
    def test_remember_and_recall(self, tmp_path, monkeypatch):
        test_file = tmp_path / "memory.json"
        monkeypatch.setattr("vox.memory._MEMORY_FILE", test_file)

        result = remember("I hate cilantro")
        assert "remember" in result.lower()

        result = recall()
        assert "cilantro" in result

    def test_forget(self, tmp_path, monkeypatch):
        test_file = tmp_path / "memory.json"
        monkeypatch.setattr("vox.memory._MEMORY_FILE", test_file)

        remember("I hate cilantro")
        result = forget("cilantro")
        assert "forgot" in result.lower()

        result = recall("cilantro")
        assert "don't have" in result.lower()

    def test_deduplication(self, tmp_path, monkeypatch):
        test_file = tmp_path / "memory.json"
        monkeypatch.setattr("vox.memory._MEMORY_FILE", test_file)

        remember("I hate cilantro")
        result = remember("I hate cilantro")
        assert "already know" in result.lower()

    def test_prompt_block(self, tmp_path, monkeypatch):
        test_file = tmp_path / "memory.json"
        monkeypatch.setattr("vox.memory._MEMORY_FILE", test_file)

        remember("favorite color is blue")
        block = build_memory_prompt_block()
        assert "blue" in block
        assert "naturally" in block.lower()


# ---------------------------------------------------------------------------
# Reminder intent detection
# ---------------------------------------------------------------------------

class TestReminderIntentDetection:
    @pytest.mark.parametrize("text,expected_minutes", [
        ("remind me in 30 minutes to take the chicken out", 30),
        ("remind me in 2 hours to call mom", 120),
        ("remind me in 5 min", 5),
    ])
    def test_remind_in_time(self, text, expected_minutes):
        result = detect_reminder_intent(text)
        assert result is not None
        assert result["action"] == "set"
        assert result["minutes"] == expected_minutes

    def test_remind_me_to(self):
        result = detect_reminder_intent("remind me to take the chicken out in 15 minutes")
        assert result is not None
        assert result["action"] == "set"
        assert result["minutes"] == 15
        assert "chicken" in result["message"]

    @pytest.mark.parametrize("text,expected_minutes", [
        ("set a timer for 5 minutes", 5),
        ("timer for 10 min", 10),
        ("set timer for 1 hour", 60),
    ])
    def test_timer(self, text, expected_minutes):
        result = detect_reminder_intent(text)
        assert result is not None
        assert result["action"] == "set"
        assert result["minutes"] == expected_minutes

    def test_list_reminders(self):
        result = detect_reminder_intent("list my reminders")
        assert result is not None
        assert result["action"] == "list"

    def test_cancel_reminder(self):
        result = detect_reminder_intent("cancel reminder about chicken")
        assert result is not None
        assert result["action"] == "cancel"

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "show me a selfie",
        "hello",
    ])
    def test_non_reminder_ignored(self, text):
        assert detect_reminder_intent(text) is None


class TestReminderStorage:
    def test_add_and_list(self, tmp_path, monkeypatch):
        test_file = tmp_path / "reminders.json"
        monkeypatch.setattr("vox.reminders._REMINDERS_FILE", test_file)

        result = add_reminder("take chicken out", 30)
        assert "30 minute" in result

        result = list_reminders()
        assert "chicken" in result

    def test_cancel(self, tmp_path, monkeypatch):
        test_file = tmp_path / "reminders.json"
        monkeypatch.setattr("vox.reminders._REMINDERS_FILE", test_file)

        add_reminder("take chicken out", 30)
        result = cancel_reminder("chicken")
        assert "cancelled" in result.lower()

        result = list_reminders()
        assert "no active" in result.lower()

"""Tests for proactive persona behavior system."""

from unittest.mock import patch
from datetime import datetime

import pytest

from vox.proactive import (
    _morning_briefing_prompt,
    _checkin_prompt,
    _goodnight_prompt,
    get_proactive_message,
    reset_daily,
)


class TestProactivePrompts:
    def test_morning_prompt_contains_day(self):
        prompt = _morning_briefing_prompt()
        assert "PROACTIVE" in prompt
        assert "morning" in prompt.lower()

    def test_checkin_prompt_is_casual(self):
        prompt = _checkin_prompt()
        assert "PROACTIVE" in prompt
        assert "1 sentence" in prompt

    def test_goodnight_prompt(self):
        prompt = _goodnight_prompt()
        assert "PROACTIVE" in prompt
        assert "goodnight" in prompt.lower()


class TestProactiveScheduling:
    def setup_method(self):
        reset_daily()

    def test_morning_triggers_between_7_and_9(self):
        """Morning briefing should have a chance of triggering 7-9 AM."""
        with patch("vox.proactive.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 5, 8, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            # Run many times — should trigger at least once with 30% chance
            triggered = False
            for _ in range(50):
                reset_daily()
                result = get_proactive_message()
                if result is not None:
                    triggered = True
                    assert "morning" in result.lower()
                    break
            assert triggered, "Morning briefing never triggered in 50 attempts"

    def test_no_trigger_at_3am(self):
        """Nothing should trigger at 3 AM."""
        with patch("vox.proactive.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 5, 3, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

            for _ in range(20):
                reset_daily()
                assert get_proactive_message() is None

    def test_morning_only_fires_once(self):
        """Morning briefing should not fire twice in the same day."""
        with patch("vox.proactive.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 5, 8, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with patch("vox.proactive.random") as mock_random:
                mock_random.random.return_value = 0.1  # Always triggers
                mock_random.choice = lambda x: x[0]

                reset_daily()
                first = get_proactive_message()
                assert first is not None

                # Second call should return None (already sent today)
                second = get_proactive_message()
                assert second is None

    def test_reset_daily_clears_state(self):
        """reset_daily should allow messages to fire again."""
        with patch("vox.proactive.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 5, 8, 0, 0)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            with patch("vox.proactive.random") as mock_random:
                mock_random.random.return_value = 0.1
                mock_random.choice = lambda x: x[0]

                reset_daily()
                first = get_proactive_message()
                assert first is not None

                reset_daily()
                second = get_proactive_message()
                assert second is not None

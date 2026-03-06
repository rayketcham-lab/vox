"""Tests for persona_life module — mood system, activities, sentiment detection."""

from __future__ import annotations

import pytest
from unittest.mock import patch
from datetime import datetime


# ---------------------------------------------------------------------------
# Mood system
# ---------------------------------------------------------------------------

class TestMoodSystem:
    def setup_method(self):
        """Reset mood to default before each test."""
        import vox.persona_life as pl
        pl._mood = 0.3

    def test_default_mood(self):
        from vox.persona_life import get_mood, get_mood_label
        assert get_mood() == 0.3
        assert get_mood_label() == "chill"

    def test_nudge_mood_positive(self):
        from vox.persona_life import nudge_mood, get_mood
        nudge_mood(0.5)
        assert get_mood() == pytest.approx(0.8)

    def test_nudge_mood_negative(self):
        from vox.persona_life import nudge_mood, get_mood
        nudge_mood(-0.5)
        assert get_mood() == pytest.approx(-0.2)

    def test_mood_clamps_high(self):
        from vox.persona_life import nudge_mood, get_mood
        nudge_mood(5.0)
        assert get_mood() == 1.0

    def test_mood_clamps_low(self):
        from vox.persona_life import nudge_mood, get_mood
        nudge_mood(-5.0)
        assert get_mood() == -1.0

    def test_mood_labels_cover_range(self):
        import vox.persona_life as pl
        # Test all label thresholds
        test_cases = [
            (-1.0, "frustrated"),
            (-0.6, "annoyed"),
            (-0.3, "meh"),
            (0.0, "neutral"),
            (0.3, "chill"),
            (0.6, "happy"),
            (0.8, "great"),
            (1.0, "amazing"),
        ]
        for mood_val, expected_label in test_cases:
            pl._mood = mood_val
            assert pl.get_mood_label() == expected_label, f"mood={mood_val}"

    def test_mood_above_max_returns_amazing(self):
        import vox.persona_life as pl
        pl._mood = 1.0
        assert pl.get_mood_label() == "amazing"


# ---------------------------------------------------------------------------
# Sentiment detection
# ---------------------------------------------------------------------------

class TestSentimentDetection:
    def test_positive_words(self):
        from vox.persona_life import detect_sentiment_nudge
        delta = detect_sentiment_nudge("thank you, that's awesome!")
        assert delta > 0

    def test_negative_words(self):
        from vox.persona_life import detect_sentiment_nudge
        delta = detect_sentiment_nudge("that's stupid and wrong")
        assert delta < 0

    def test_neutral_message(self):
        from vox.persona_life import detect_sentiment_nudge
        delta = detect_sentiment_nudge("what time is it?")
        assert delta == 0.0

    def test_clamped_to_range(self):
        from vox.persona_life import detect_sentiment_nudge
        # Even with many positive words, should clamp to 0.3
        delta = detect_sentiment_nudge(
            "thank you love awesome great nice cool haha lol beautiful perfect amazing wow"
        )
        assert delta <= 0.3

    def test_negative_clamped(self):
        from vox.persona_life import detect_sentiment_nudge
        delta = detect_sentiment_nudge(
            "hate stupid wrong bad annoying shut up terrible awful boring whatever ugh"
        )
        assert delta >= -0.3

    def test_mixed_sentiment(self):
        from vox.persona_life import detect_sentiment_nudge
        delta = detect_sentiment_nudge("that's awesome but also kind of annoying")
        # Should be small — positive and negative cancel partially
        assert abs(delta) < 0.2


# ---------------------------------------------------------------------------
# Activity system
# ---------------------------------------------------------------------------

class TestActivitySystem:
    def setup_method(self):
        import vox.persona_life as pl
        pl._current_activity = None
        pl._activity_set_hour = -1

    def test_get_activity_returns_string(self):
        from vox.persona_life import get_activity
        activity = get_activity()
        assert isinstance(activity, str)
        assert len(activity) > 0

    def test_activity_stays_same_within_hour(self):
        from vox.persona_life import get_activity
        a1 = get_activity()
        a2 = get_activity()
        assert a1 == a2

    def test_force_activity(self):
        from vox.persona_life import force_activity, get_activity
        force_activity("coding with Ray")
        assert get_activity() == "coding with Ray"

    @patch("vox.persona_life.datetime")
    def test_morning_activity(self, mock_dt):
        import vox.persona_life as pl
        pl._current_activity = None
        pl._activity_set_hour = -1
        mock_dt.now.return_value = datetime(2026, 3, 5, 8, 0)
        activity = pl.get_activity()
        assert isinstance(activity, str)
        assert len(activity) > 0

    @patch("vox.persona_life.datetime")
    def test_late_night_activity(self, mock_dt):
        import vox.persona_life as pl
        pl._current_activity = None
        pl._activity_set_hour = -1
        mock_dt.now.return_value = datetime(2026, 3, 5, 2, 0)
        activity = pl.get_activity()
        assert isinstance(activity, str)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class TestBuildLifeContext:
    def test_returns_string(self):
        from vox.persona_life import build_life_context
        ctx = build_life_context()
        assert isinstance(ctx, str)

    def test_contains_mood(self):
        from vox.persona_life import build_life_context
        ctx = build_life_context()
        assert "Mood:" in ctx

    def test_contains_activity(self):
        from vox.persona_life import build_life_context
        ctx = build_life_context()
        assert "Activity:" in ctx

    def test_contains_time_of_day(self):
        from vox.persona_life import build_life_context
        ctx = build_life_context()
        assert "Time:" in ctx

    def test_mood_reflects_nudge(self):
        import vox.persona_life as pl
        pl._mood = 0.3
        pl.nudge_mood(0.6)  # Should be "great" now
        ctx = pl.build_life_context()
        assert "great" in ctx or "amazing" in ctx

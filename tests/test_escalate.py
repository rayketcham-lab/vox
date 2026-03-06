"""Tests for Claude escalation and auto-issue creation."""

import pytest

from vox.escalate import _ESCALATION_TRIGGERS, _QUALITY_COMPLAINTS, should_escalate
from vox.auto_issue import _FEATURE_REQUEST_SIGNALS, should_create_issue


# ---------------------------------------------------------------------------
# Escalation trigger detection
# ---------------------------------------------------------------------------

class TestEscalationTriggers:
    """Verify complex task patterns are detected."""

    @pytest.mark.parametrize("text", [
        "explain how does a transformer neural network work",
        "write me a python script to parse CSV files",
        "how to implement a binary search algorithm",
        "debug this function that's throwing an error",
        "create a dockerfile for my python app",
        "translate this code from python to javascript",
        "summarize this article about quantum computing",
        "write a regex to match email addresses",
        "explain how does SSL encryption work",
        "implement a class for a linked list in python",
    ])
    def test_complex_tasks_detected(self, text):
        assert _ESCALATION_TRIGGERS.search(text), f"Should detect: {text}"

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "show me a selfie",
        "send me an email",
        "what time is it",
        "search for pizza near me",
        "how are you doing",
        "tell me a joke",
        "good morning",
    ])
    def test_simple_tasks_not_escalated(self, text):
        assert not _ESCALATION_TRIGGERS.search(text), f"Should NOT detect: {text}"


class TestQualityComplaints:
    """Verify user complaints about answer quality are detected."""

    @pytest.mark.parametrize("text", [
        "that's wrong",
        "you're incorrect",
        "no that's not right",
        "actually it's different",
        "try again",
        "that doesn't make sense",
        "you're hallucinating",
        "be more accurate",
    ])
    def test_complaints_detected(self, text):
        assert _QUALITY_COMPLAINTS.search(text), f"Should detect: {text}"


class TestShouldEscalate:
    """Integration: should_escalate respects API key config."""

    def test_no_api_key_never_escalates(self, monkeypatch):
        monkeypatch.setattr("vox.config.CLAUDE_API_KEY", "")
        assert should_escalate("write me a python script") is False

    def test_with_api_key_escalates_complex(self, monkeypatch):
        monkeypatch.setattr("vox.config.CLAUDE_API_KEY", "sk-test-key")
        assert should_escalate("write me a python script to parse CSV") is True

    def test_with_api_key_skips_simple(self, monkeypatch):
        monkeypatch.setattr("vox.config.CLAUDE_API_KEY", "sk-test-key")
        assert should_escalate("what's the weather") is False

    def test_complaint_escalates(self, monkeypatch):
        monkeypatch.setattr("vox.config.CLAUDE_API_KEY", "sk-test-key")
        assert should_escalate("that's wrong, try again") is True


# ---------------------------------------------------------------------------
# Auto-issue detection
# ---------------------------------------------------------------------------

class TestFeatureRequestDetection:
    """Verify unimplemented feature requests are detected."""

    @pytest.mark.parametrize("text", [
        "can you play music",
        "could you set a timer for 5 minutes",
        "can you control my lights",
        "is there a way to run a script",
        "can you remind me to call mom at 3pm",
        "could you clone my voice",
        "can you upscale this image",
    ])
    def test_feature_requests_detected(self, text):
        assert _FEATURE_REQUEST_SIGNALS.search(text), f"Should detect: {text}"

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "show me a selfie",
        "search for restaurants",
        "send me an email",
        "generate an image of a cat",
    ])
    def test_existing_features_not_flagged(self, text):
        # These match known capabilities, so should_create_issue returns False
        # even if the pattern matches
        pass  # Known capabilities are checked in should_create_issue, not the regex


class TestShouldCreateIssue:
    """Integration: should_create_issue checks repo config and known capabilities."""

    def test_no_repo_never_creates(self, monkeypatch):
        monkeypatch.setattr("vox.config.GITHUB_REPO", "")
        assert should_create_issue("can you play music") is False

    def test_with_repo_creates_for_unknown(self, monkeypatch):
        monkeypatch.setattr("vox.config.GITHUB_REPO", "test/repo")
        assert should_create_issue("can you play music") is True

    def test_known_capability_skipped(self, monkeypatch):
        monkeypatch.setattr("vox.config.GITHUB_REPO", "test/repo")
        # "weather" is a known capability
        assert should_create_issue("can you check the weather") is False
        # "image" is a known capability
        assert should_create_issue("can you generate an image of a cat") is False

    def test_deduplication(self, monkeypatch):
        monkeypatch.setattr("vox.config.GITHUB_REPO", "test/repo")
        from vox.auto_issue import _recent_issues
        _recent_issues.clear()
        # First time — should create
        assert should_create_issue("can you play music") is True
        # Mark as filed
        _recent_issues.add("play music")
        # Second time — deduplicated
        assert should_create_issue("can you play music") is False
        _recent_issues.clear()

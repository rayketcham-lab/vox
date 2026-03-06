"""Tests for conversation logging."""


import pytest

from vox.conversation_log import get_recent, log_exchange, search


@pytest.fixture(autouse=True)
def _temp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.conversation_log._LOG_DIR", tmp_path)


def test_log_and_retrieve():
    log_exchange("hello", "hi there!")
    entries = get_recent(days=1)
    assert len(entries) == 1
    assert entries[0]["user"] == "hello"
    assert entries[0]["assistant"] == "hi there!"


def test_log_with_tools():
    log_exchange("what's the weather?", "sunny and 72F", tools_used=["get_weather"])
    entries = get_recent(days=1)
    assert entries[0]["tools"] == ["get_weather"]


def test_multiple_entries():
    log_exchange("first", "reply1")
    log_exchange("second", "reply2")
    entries = get_recent(days=1, limit=10)
    assert len(entries) == 2
    # Most recent first
    assert entries[0]["user"] == "second"


def test_search_by_keyword():
    log_exchange("tell me about Python", "Python is a programming language")
    log_exchange("what's the weather?", "sunny")
    results = search("Python")
    assert len(results) == 1
    assert "Python" in results[0]["user"]


def test_empty_log():
    entries = get_recent(days=1)
    assert entries == []


def test_search_no_results():
    log_exchange("hello", "hi")
    results = search("nonexistent")
    assert results == []

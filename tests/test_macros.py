"""Tests for voice macros."""

import pytest

from vox.macros import add_macro, execute_macro, find_macro, list_macros, remove_macro


@pytest.fixture(autouse=True)
def _temp_macros(tmp_path, monkeypatch):
    monkeypatch.setattr("vox.macros._MACROS_FILE", tmp_path / "macros.json")


def test_add_and_list():
    add_macro("morning briefing", [
        {"tool": "get_weather", "args": {}},
        {"tool": "list_notes", "args": {}},
    ])
    macros = list_macros()
    assert len(macros) == 1
    assert macros[0]["name"] == "morning briefing"
    assert len(macros[0]["steps"]) == 2


def test_find_exact():
    add_macro("morning briefing", [{"tool": "get_weather", "args": {}}])
    macro = find_macro("morning briefing")
    assert macro is not None
    assert macro["name"] == "morning briefing"


def test_find_substring():
    add_macro("morning briefing", [{"tool": "get_weather", "args": {}}])
    macro = find_macro("run my morning briefing please")
    assert macro is not None


def test_find_no_match():
    assert find_macro("nonexistent") is None


def test_remove():
    add_macro("test macro", [{"tool": "get_current_time", "args": {}}])
    assert remove_macro("test macro") is True
    assert list_macros() == []


def test_remove_nonexistent():
    assert remove_macro("nope") is False


def test_execute():
    add_macro("time check", [{"tool": "get_current_time", "args": {}}])
    macro = find_macro("time check")
    results = execute_macro(macro)
    assert len(results) == 1
    assert results[0][0] == "get_current_time"
    assert len(results[0][1]) > 0


def test_empty_list():
    assert list_macros() == []

"""Tests for multi-step task planner."""

from vox.planner import build_plan_prompt, format_plan_for_user, is_multi_step


def test_multi_step_search_then_email():
    assert is_multi_step("search for oil filters and then email me the results")


def test_multi_step_research_and_send():
    assert is_multi_step("research the best GPU prices and email me a summary")


def test_multi_step_compare_and_save():
    assert is_multi_step("compare these products then save a note with the best one")


def test_simple_request_not_multi_step():
    assert not is_multi_step("what's the weather?")


def test_simple_search_not_multi_step():
    assert not is_multi_step("search for python tutorials")


def test_greeting_not_multi_step():
    assert not is_multi_step("hello how are you")


def test_build_plan_prompt():
    prompt = build_plan_prompt("search for oil filters and email me")
    assert "oil filters" in prompt
    assert "web_search" in prompt


def test_format_plan_numbered():
    plan = "1. [web_search] — search for filters\n2. [send_email] — email results"
    formatted = format_plan_for_user(plan)
    assert "1." in formatted
    assert "2." in formatted


def test_format_plan_empty():
    assert len(format_plan_for_user("")) > 0

"""Tests for tool execution."""

from vox.tools import execute_tool


def test_get_current_time():
    result = execute_tool("get_current_time", {})
    assert isinstance(result, str)
    assert len(result) > 0


def test_unknown_tool():
    result = execute_tool("nonexistent", {})
    assert "Unknown tool" in result

"""Tests for todo list system."""

import pytest

from vox.todos import (
    add_todo,
    complete_todo,
    detect_todo_intent,
    list_todos,
    remove_todo,
)


class TestTodoIntentDetection:
    @pytest.mark.parametrize("text", [
        "add buy milk to my todo list",
        "put call dentist on my task list",
        "todo: pick up dry cleaning",
    ])
    def test_add_detected(self, text):
        result = detect_todo_intent(text)
        assert result is not None
        assert result["action"] == "add"
        assert len(result["task"]) > 2

    @pytest.mark.parametrize("text", [
        "show my todo list",
        "what's on my task list",
        "list my todos",
        "my todo list",
    ])
    def test_list_detected(self, text):
        result = detect_todo_intent(text)
        assert result is not None
        assert result["action"] == "list"

    @pytest.mark.parametrize("text", [
        "mark buy milk as done",
        "finished with the dentist todo",
        "done with laundry task",
    ])
    def test_complete_detected(self, text):
        result = detect_todo_intent(text)
        assert result is not None
        assert result["action"] == "complete"

    @pytest.mark.parametrize("text", [
        "what's the weather",
        "show me a selfie",
        "hello how are you",
    ])
    def test_non_todo_ignored(self, text):
        assert detect_todo_intent(text) is None


class TestTodoStorage:
    def test_add_and_list(self, tmp_path, monkeypatch):
        test_file = tmp_path / "todos.json"
        monkeypatch.setattr("vox.todos._TODOS_FILE", test_file)

        result = add_todo("buy milk")
        assert "Added" in result

        result = list_todos()
        assert "milk" in result

    def test_complete(self, tmp_path, monkeypatch):
        test_file = tmp_path / "todos.json"
        monkeypatch.setattr("vox.todos._TODOS_FILE", test_file)

        add_todo("buy milk")
        result = complete_todo("milk")
        assert "Done" in result

        result = list_todos()
        assert "empty" in result.lower()

    def test_deduplication(self, tmp_path, monkeypatch):
        test_file = tmp_path / "todos.json"
        monkeypatch.setattr("vox.todos._TODOS_FILE", test_file)

        add_todo("buy milk")
        result = add_todo("buy milk")
        assert "already" in result.lower()

    def test_remove(self, tmp_path, monkeypatch):
        test_file = tmp_path / "todos.json"
        monkeypatch.setattr("vox.todos._TODOS_FILE", test_file)

        add_todo("buy milk")
        result = remove_todo("milk")
        assert "Removed" in result

        result = list_todos()
        assert "empty" in result.lower()

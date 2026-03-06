"""Todo list — persistent task tracking via natural language.

Supports:
- "add X to my todo list" / "I need to X"
- "what's on my list" / "show my todos"
- "mark X as done" / "finished X"
- "remove X from my list"

Storage: data/todos.json (gitignored)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_TODOS_FILE = Path(__file__).parent.parent.parent / "data" / "todos.json"


def _load() -> list[dict]:
    if not _TODOS_FILE.exists():
        return []
    try:
        return json.loads(_TODOS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(todos: list[dict]) -> None:
    _TODOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TODOS_FILE.write_text(json.dumps(todos, indent=2, default=str), encoding="utf-8")


def add_todo(task: str) -> str:
    """Add a task to the todo list."""
    todos = _load()
    # Dedup check
    for t in todos:
        if t["task"].lower() == task.lower() and not t.get("done"):
            return f"'{task}' is already on your list."
    todos.append({
        "task": task,
        "created": datetime.now().isoformat(),
        "done": False,
    })
    _save(todos)
    active = [t for t in todos if not t.get("done")]
    return f"Added: {task} ({len(active)} items on your list)"


def list_todos() -> str:
    """List active todos."""
    todos = _load()
    active = [t for t in todos if not t.get("done")]
    if not active:
        return "Your todo list is empty."
    lines = []
    for i, t in enumerate(active, 1):
        lines.append(f"{i}. {t['task']}")
    return "Your todo list:\n" + "\n".join(lines)


def complete_todo(keyword: str) -> str:
    """Mark a todo as done by keyword match."""
    todos = _load()
    completed = []
    for t in todos:
        if not t.get("done") and keyword.lower() in t["task"].lower():
            t["done"] = True
            t["completed"] = datetime.now().isoformat()
            completed.append(t["task"])
    if not completed:
        return f"No active todo matches '{keyword}'."
    _save(todos)
    # Clean up old completed items (keep last 50)
    active = [t for t in todos if not t.get("done")]
    done = [t for t in todos if t.get("done")]
    _save(active + done[-50:])
    return f"Done: {', '.join(completed)}"


def remove_todo(keyword: str) -> str:
    """Remove a todo by keyword match."""
    todos = _load()
    before = len(todos)
    todos = [t for t in todos if keyword.lower() not in t["task"].lower() or t.get("done")]
    after = len(todos)
    removed = before - after
    if removed == 0:
        return f"No active todo matches '{keyword}'."
    _save(todos)
    return f"Removed {removed} item{'s' if removed > 1 else ''}."


# --- Intent detection ---

_ADD_TODO = re.compile(
    r"(?:add|put)\s+(.+?)(?:\s+(?:to|on)\s+(?:my\s+)?(?:todo|to-do|task)\s*(?:list)?)"
    r"|(?:(?:todo|to-do)\s*:\s+(.+))",
    re.IGNORECASE,
)

_LIST_TODOS = re.compile(
    r"\b(?:show|list|what(?:'s| is))\b.*\b(?:todo|to-do|task)\b"
    r"|\b(?:todo|to-do|task)\s*list\b"
    r"|\bmy\s+(?:todo|to-do|task)s?\b",
    re.IGNORECASE,
)

_COMPLETE_TODO = re.compile(
    r"\b(?:done|finish(?:ed)?|complete(?:d)?|check\s+off)\b.*?(?:with\s+|about\s+)?(.+)"
    r"|\bmark\s+(.+?)\s+(?:as\s+)?(?:done|complete|finished)\b",
    re.IGNORECASE,
)

_REMOVE_TODO = re.compile(
    r"\b(?:remove|delete)\b.*\b(?:todo|to-do|task)\b.*?(?:about\s+)?(.+)?$",
    re.IGNORECASE,
)


def detect_todo_intent(text: str) -> dict | None:
    """Check if the user wants to add/list/complete/remove a todo.

    Returns {"action": "add"|"list"|"complete"|"remove", ...} or None.
    """
    # Add first — "add X to my todo list" contains "todo list" which would match list
    m = _ADD_TODO.search(text)
    if m:
        task = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if task and len(task) > 2:
            return {"action": "add", "task": task}

    # Remove (before list — "remove X from todo list" also contains "todo list")
    m = _REMOVE_TODO.search(text)
    if m:
        keyword = (m.group(1) or "").strip()
        return {"action": "remove", "keyword": keyword or "all"}

    # Complete
    m = _COMPLETE_TODO.search(text)
    if m and any(w in text.lower() for w in ("todo", "to-do", "task", "done", "finished", "completed")):
        keyword = (m.group(1) or m.group(2) or "").strip()
        if keyword:
            return {"action": "complete", "keyword": keyword}

    # List (last — broadest pattern)
    if _LIST_TODOS.search(text):
        return {"action": "list"}

    return None

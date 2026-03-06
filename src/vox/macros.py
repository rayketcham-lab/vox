"""Voice-activated macros — user-defined tool chains.

Macros are saved sequences of tool calls that can be triggered by a phrase.
Stored in data/macros.json. Each macro has a trigger phrase and an ordered
list of steps (tool name + args).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_MACROS_FILE: Path | None = None


def _get_file() -> Path:
    global _MACROS_FILE
    if _MACROS_FILE is None:
        from vox.config import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        _MACROS_FILE = data_dir / "macros.json"
    return _MACROS_FILE


def _load() -> dict[str, dict]:
    path = _get_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save(macros: dict[str, dict]):
    path = _get_file()
    path.write_text(json.dumps(macros, indent=2), encoding="utf-8")


def add_macro(name: str, steps: list[dict]) -> dict:
    """Add or update a macro.

    Args:
        name: trigger phrase (e.g., "morning briefing")
        steps: list of {"tool": "tool_name", "args": {}} dicts
    """
    macros = _load()
    macro = {"name": name, "steps": steps}
    macros[name.lower()] = macro
    _save(macros)
    log.info("Macro saved: '%s' with %d steps", name, len(steps))
    return macro


def remove_macro(name: str) -> bool:
    """Remove a macro by name."""
    macros = _load()
    key = name.lower()
    if key in macros:
        del macros[key]
        _save(macros)
        return True
    return False


def list_macros() -> list[dict]:
    """Return all defined macros."""
    return list(_load().values())


def find_macro(text: str) -> dict | None:
    """Find a macro whose trigger phrase matches the text."""
    macros = _load()
    text_lower = text.lower().strip()
    # Exact match first
    if text_lower in macros:
        return macros[text_lower]
    # Substring match
    for key, macro in macros.items():
        if key in text_lower:
            return macro
    return None


def execute_macro(macro: dict) -> list[tuple[str, str]]:
    """Execute a macro's steps sequentially, returning (tool_name, result) pairs."""
    from vox.tools import execute_tool

    results = []
    for step in macro.get("steps", []):
        tool = step.get("tool", "")
        args = dict(step.get("args", {}))
        log.info("Macro step: %s(%s)", tool, args)
        result = execute_tool(tool, args)
        results.append((tool, result))
    return results

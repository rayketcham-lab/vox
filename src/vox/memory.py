"""Persistent user memory — Ann remembers things across sessions.

Stores facts the user tells her (preferences, dates, names, etc.) in a
JSON file. Facts are injected into the system prompt so the LLM can
reference them naturally.

Storage: data/user_memory.json (gitignored)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_MEMORY_FILE = Path(__file__).parent.parent.parent / "data" / "user_memory.json"
_MAX_FACTS = 50  # keep prompt injection reasonable


def _load() -> list[dict]:
    """Load memory facts from disk."""
    if not _MEMORY_FILE.exists():
        return []
    try:
        return json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        log.warning("Failed to load memory file, starting fresh")
        return []


def _save(facts: list[dict]) -> None:
    """Save memory facts to disk."""
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _MEMORY_FILE.write_text(json.dumps(facts, indent=2), encoding="utf-8")


def remember(fact: str, category: str = "general") -> str:
    """Store a fact about the user. Returns confirmation message."""
    facts = _load()

    # Deduplicate — don't store the same fact twice
    for existing in facts:
        if existing["fact"].lower().strip() == fact.lower().strip():
            return f"I already know that: {fact}"

    facts.append({
        "fact": fact,
        "category": category,
        "when": datetime.now().isoformat(),
    })

    # Trim oldest if over limit
    if len(facts) > _MAX_FACTS:
        facts = facts[-_MAX_FACTS:]

    _save(facts)
    log.info("Remembered: [%s] %s", category, fact)
    return "Got it, I'll remember that."


def forget(keyword: str) -> str:
    """Forget facts matching a keyword."""
    facts = _load()
    before = len(facts)
    facts = [f for f in facts if keyword.lower() not in f["fact"].lower()]
    after = len(facts)
    if before == after:
        return f"I don't have anything about '{keyword}' to forget."
    _save(facts)
    removed = before - after
    return f"Done — forgot {removed} thing{'s' if removed > 1 else ''} about '{keyword}'."


def recall(keyword: str = "") -> str:
    """Recall facts, optionally filtered by keyword."""
    facts = _load()
    if not facts:
        return "I don't have anything stored yet."
    if keyword:
        facts = [f for f in facts if keyword.lower() in f["fact"].lower()]
        if not facts:
            return f"I don't have anything about '{keyword}'."
    lines = [f"- {f['fact']}" for f in facts[-20:]]
    return "Here's what I remember:\n" + "\n".join(lines)


def build_memory_prompt_block() -> str:
    """Build a prompt block with all stored facts for system prompt injection."""
    facts = _load()
    if not facts:
        return ""
    # Group by category
    lines = [f"- {f['fact']}" for f in facts[-30:]]
    return (
        "\nThings you know about the user (reference naturally, don't list them):\n"
        + "\n".join(lines)
    )


# --- Intent detection for memory commands ---

_REMEMBER_PATTERN = re.compile(
    r"\b(remember|don'?t forget|keep in mind|note that|fyi)\b[:\s]+(.+)",
    re.IGNORECASE,
)

_FORGET_PATTERN = re.compile(
    r"\b(forget|stop remembering|remove|delete)\b.*\babout\b\s+(.+)",
    re.IGNORECASE,
)

_RECALL_PATTERN = re.compile(
    r"\b(what do you (know|remember)|recall|what did i tell you)\b",
    re.IGNORECASE,
)


def detect_memory_intent(text: str) -> dict | None:
    """Check if the user wants to store/recall/forget a memory.

    Returns {"action": "remember"|"forget"|"recall", "data": str} or None.
    """
    m = _REMEMBER_PATTERN.search(text)
    if m:
        fact = m.group(2).strip().rstrip(".")
        if len(fact) > 5:  # ignore tiny fragments
            return {"action": "remember", "data": fact}

    m = _FORGET_PATTERN.search(text)
    if m:
        return {"action": "forget", "data": m.group(2).strip()}

    if _RECALL_PATTERN.search(text):
        return {"action": "recall", "data": ""}

    return None

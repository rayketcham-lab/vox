"""Reminders and timers — persistent reminders with WebSocket push notifications.

Supports:
- Relative timers: "remind me in 30 minutes"
- Named reminders: "remind me to take the chicken out"
- List/cancel reminders

Storage: data/reminders.json (gitignored)
Background loop checks every 30 seconds and fires due reminders via WebSocket.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_REMINDERS_FILE = Path(__file__).parent.parent.parent / "data" / "reminders.json"

# Active reminder callbacks (set by the web server's background loop)
_notify_callbacks: list[callable] = []


def register_notify(callback: callable) -> None:
    """Register a callback to be called when a reminder fires.

    Callback signature: callback(message: str) -> None
    """
    _notify_callbacks.append(callback)


def _load() -> list[dict]:
    if not _REMINDERS_FILE.exists():
        return []
    try:
        return json.loads(_REMINDERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save(reminders: list[dict]) -> None:
    _REMINDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _REMINDERS_FILE.write_text(json.dumps(reminders, indent=2, default=str), encoding="utf-8")


def add_reminder(message: str, minutes: int) -> str:
    """Add a reminder that fires in N minutes."""
    reminders = _load()
    due_at = datetime.now() + timedelta(minutes=minutes)
    reminders.append({
        "message": message,
        "due_at": due_at.isoformat(),
        "created": datetime.now().isoformat(),
        "fired": False,
    })
    _save(reminders)
    if minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        time_str = f"{hours}h{mins}m" if mins else f"{hours} hour{'s' if hours > 1 else ''}"
    else:
        time_str = f"{minutes} minute{'s' if minutes > 1 else ''}"
    log.info("Reminder set: '%s' in %s (due %s)", message, time_str, due_at.strftime("%I:%M %p"))
    return f"Got it — I'll remind you in {time_str}: {message}"


def list_reminders() -> str:
    """List active (unfired) reminders."""
    reminders = _load()
    active = [r for r in reminders if not r.get("fired")]
    if not active:
        return "No active reminders."
    lines = []
    for i, r in enumerate(active, 1):
        due = datetime.fromisoformat(r["due_at"])
        remaining = due - datetime.now()
        if remaining.total_seconds() > 0:
            mins = int(remaining.total_seconds() // 60)
            lines.append(f"{i}. {r['message']} — in {mins} min")
        else:
            lines.append(f"{i}. {r['message']} — overdue!")
    return "Active reminders:\n" + "\n".join(lines)


def cancel_reminder(keyword: str) -> str:
    """Cancel reminders matching a keyword."""
    reminders = _load()
    before = len([r for r in reminders if not r.get("fired")])
    reminders = [r for r in reminders if keyword.lower() not in r["message"].lower() or r.get("fired")]
    after = len([r for r in reminders if not r.get("fired")])
    _save(reminders)
    removed = before - after
    if removed == 0:
        return f"No active reminders match '{keyword}'."
    return f"Cancelled {removed} reminder{'s' if removed > 1 else ''}."


def check_and_fire() -> list[str]:
    """Check for due reminders and fire them. Returns list of fired messages."""
    reminders = _load()
    now = datetime.now()
    fired = []
    changed = False

    for r in reminders:
        if r.get("fired"):
            continue
        due = datetime.fromisoformat(r["due_at"])
        if now >= due:
            r["fired"] = True
            changed = True
            msg = f"Reminder: {r['message']}"
            fired.append(msg)
            log.info("Reminder fired: %s", r["message"])
            for cb in _notify_callbacks:
                try:
                    cb(msg)
                except Exception:
                    log.exception("Reminder notify callback failed")

    if changed:
        # Clean up old fired reminders (keep last 20)
        unfired = [r for r in reminders if not r.get("fired")]
        old_fired = [r for r in reminders if r.get("fired")]
        _save(unfired + old_fired[-20:])

    return fired


# --- Intent detection for reminder commands ---

_REMINDER_PATTERN = re.compile(
    r"\bremind\s+me\b.*?\bin\s+(\d+)\s*(min(?:ute)?s?|hours?|hr?s?)\b"
    r"(?:\s*(?:to|about|that)\s+(.+))?",
    re.IGNORECASE,
)

_REMINDER_TO_PATTERN = re.compile(
    r"\bremind\s+me\s+to\s+(.+?)(?:\s+in\s+(\d+)\s*(min(?:ute)?s?|hours?|hr?s?))?$",
    re.IGNORECASE,
)

_TIMER_PATTERN = re.compile(
    r"\b(?:set\s+a?\s*)?timer\s+(?:for\s+)?(\d+)\s*(min(?:ute)?s?|hours?|hr?s?|seconds?|secs?)\b",
    re.IGNORECASE,
)

_LIST_REMINDERS = re.compile(
    r"\b(list|show|what|check)\b.*\breminders?\b"
    r"|\breminders?\s+list\b",
    re.IGNORECASE,
)

_CANCEL_REMINDER = re.compile(
    r"\b(cancel|remove|delete|clear)\b.*\breminders?\b.*?(?:about\s+)?(.+)?$",
    re.IGNORECASE,
)


def _parse_minutes(amount: str, unit: str) -> int:
    """Convert amount + unit to minutes."""
    n = int(amount)
    unit = unit.lower()
    if unit.startswith("h"):
        return n * 60
    if unit.startswith("s"):
        return max(1, n // 60)  # seconds → at least 1 minute
    return n


def detect_reminder_intent(text: str) -> dict | None:
    """Check if the user wants to set/list/cancel a reminder.

    Returns {"action": "set"|"list"|"cancel", ...} or None.
    """
    # "remind me to take the chicken out in 30 minutes" (check first — more specific)
    m = _REMINDER_TO_PATTERN.search(text)
    if m:
        message = m.group(1).strip()
        if m.group(2) and m.group(3):
            minutes = _parse_minutes(m.group(2), m.group(3))
        else:
            minutes = 30  # default 30 min if no time specified
        return {"action": "set", "message": message, "minutes": minutes}

    # "remind me in 30 minutes to take chicken out"
    m = _REMINDER_PATTERN.search(text)
    if m:
        minutes = _parse_minutes(m.group(1), m.group(2))
        message = (m.group(3) or "").strip() or "Time's up!"
        return {"action": "set", "message": message, "minutes": minutes}

    # "set a timer for 5 minutes"
    m = _TIMER_PATTERN.search(text)
    if m:
        minutes = _parse_minutes(m.group(1), m.group(2))
        return {"action": "set", "message": "Timer done!", "minutes": minutes}

    # "list reminders" / "what reminders do I have"
    if _LIST_REMINDERS.search(text):
        return {"action": "list"}

    # "cancel reminder about chicken"
    m = _CANCEL_REMINDER.search(text)
    if m:
        keyword = (m.group(2) or "").strip()
        return {"action": "cancel", "keyword": keyword or "all"}

    return None

"""User preference learning — captures corrections and applies them as rules.

When the user corrects VOX ("I didn't ask you to email that", "always show
images in chat", "don't call me sir"), the correction is stored and injected
into future system prompts so the same mistake doesn't happen twice.

Preferences persist in persona/preferences.yaml (gitignored).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

import yaml

from vox.config import PROJECT_ROOT

log = logging.getLogger(__name__)

PREFS_FILE = PROJECT_ROOT / "persona" / "preferences.yaml"

# In-memory cache
_preferences: dict = {}

# Patterns that suggest the user is correcting behavior
_CORRECTION_PATTERNS = [
    re.compile(
        r"(?:i\s+)?(?:didn.t|did\s+not|don.t|do\s+not|dont)\s+(?:ask|want|need)\s+(?:you\s+)?(?:to\s+)?(.+)", re.I,
    ),
    re.compile(r"(?:dont|don.t|do\s+not)\s+(.+?)(?:\s+unless\b.+)?$", re.I),
    re.compile(r"(?:stop|quit|never)\s+(.+)", re.I),
    re.compile(r"(?:always|from\s+now\s+on)\s+(.+)", re.I),
    re.compile(r"(?:instead|rather)\s+(.+)", re.I),
    re.compile(r"(?:next\s+time|in\s+the\s+future)\s+(.+)", re.I),
    re.compile(r"(?:remember\s+(?:that|to))\s+(.+)", re.I),
    re.compile(r"(?:don.t\s+call\s+me|call\s+me)\s+(.+)", re.I),
    re.compile(r"(?:i\s+prefer)\s+(.+)", re.I),
]


def load_preferences() -> dict:
    """Load preferences from disk."""
    global _preferences
    if PREFS_FILE.exists():
        with open(PREFS_FILE, encoding="utf-8") as f:
            _preferences = yaml.safe_load(f) or {}
    else:
        _preferences = {"rules": [], "corrections": []}
    return _preferences


def save_preferences() -> None:
    """Save preferences to disk."""
    PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        yaml.dump(_preferences, f, default_flow_style=False, allow_unicode=True)
    log.info("Preferences saved: %d rules, %d corrections",
             len(_preferences.get("rules", [])), len(_preferences.get("corrections", [])))


def detect_correction(user_message: str) -> str | None:
    """Check if the user's message is a correction/preference statement.

    Returns the extracted rule text, or None if not a correction.
    """
    for pattern in _CORRECTION_PATTERNS:
        match = pattern.search(user_message)
        if match:
            return match.group(0).strip()
    return None


def add_rule(rule_text: str, source_message: str = "") -> dict:
    """Add a learned rule from a user correction.

    Returns the new rule dict.
    """
    if "rules" not in _preferences:
        _preferences["rules"] = []

    # Check for duplicate
    for existing in _preferences["rules"]:
        if existing.get("rule", "").lower() == rule_text.lower():
            log.info("Rule already exists: %s", rule_text)
            return existing

    rule = {
        "rule": rule_text,
        "source": source_message[:200] if source_message else "",
        "added": datetime.now().isoformat(),
    }
    _preferences["rules"].append(rule)

    # Also keep a log of corrections for context
    if "corrections" not in _preferences:
        _preferences["corrections"] = []
    _preferences["corrections"].append({
        "message": source_message[:200],
        "rule_added": rule_text,
        "when": datetime.now().isoformat(),
    })

    save_preferences()
    log.info("New rule learned: %s", rule_text)
    return rule


def add_manual_rule(rule_text: str) -> dict:
    """Add a manual rule (user explicitly says 'remember X')."""
    return add_rule(rule_text, source_message="manual")


def remove_rule(rule_text: str) -> bool:
    """Remove a rule by text (partial match)."""
    rules = _preferences.get("rules", [])
    for i, r in enumerate(rules):
        if rule_text.lower() in r.get("rule", "").lower():
            removed = rules.pop(i)
            save_preferences()
            log.info("Rule removed: %s", removed["rule"])
            return True
    return False


def get_rules() -> list[str]:
    """Get all active rules as strings for prompt injection."""
    return [r["rule"] for r in _preferences.get("rules", [])]


def build_preferences_block() -> str:
    """Build the preferences block for the system prompt.

    Returns empty string if no rules exist.
    """
    rules = get_rules()
    if not rules:
        return ""
    lines = "\n".join(f"- {r}" for r in rules)
    return f"\nUSER PREFERENCES (learned from past corrections — follow these):\n{lines}\n"


# Auto-load on import
load_preferences()

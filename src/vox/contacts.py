"""Local contacts / address book with fuzzy name matching.

JSON-backed contacts store. Supports names, emails, phones, tags/groups,
and fuzzy lookup so the user can say "email John" without remembering
John's last name or email address.
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

log = logging.getLogger(__name__)

_CONTACTS_FILE: Path | None = None


def _get_file() -> Path:
    global _CONTACTS_FILE
    if _CONTACTS_FILE is None:
        from vox.config import PROJECT_ROOT
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        _CONTACTS_FILE = data_dir / "contacts.json"
    return _CONTACTS_FILE


def _load() -> list[dict]:
    path = _get_file()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _save(contacts: list[dict]):
    path = _get_file()
    path.write_text(json.dumps(contacts, indent=2, default=str), encoding="utf-8")


def _next_id(contacts: list[dict]) -> int:
    return max((c.get("id", 0) for c in contacts), default=0) + 1


def _fuzzy_score(query: str, name: str) -> float:
    """Score how well query matches name (0-1). Case-insensitive."""
    q, n = query.lower(), name.lower()
    # Exact substring match scores high
    if q in n:
        return 0.95
    # Check each word in the name
    best = 0.0
    for part in n.split():
        if part.startswith(q):
            return 0.9
        # Per-word similarity (catches nicknames like Mike/Michael)
        word_score = SequenceMatcher(None, q, part).ratio()
        best = max(best, word_score)
    # Blend: best word match weighted higher than full-name match
    full_score = SequenceMatcher(None, q, n).ratio()
    return max(best, full_score)


def lookup(query: str, threshold: float = 0.5) -> list[dict]:
    """Find contacts matching a name query (fuzzy)."""
    contacts = _load()
    scored = []
    for c in contacts:
        name = c.get("name", "")
        score = _fuzzy_score(query, name)
        # Also check tags/groups
        for tag in c.get("tags", []):
            tag_score = _fuzzy_score(query, tag)
            score = max(score, tag_score)
        if score >= threshold:
            scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


def lookup_group(group: str) -> list[dict]:
    """Find all contacts in a tag/group."""
    contacts = _load()
    group_lower = group.lower()
    return [c for c in contacts if group_lower in [t.lower() for t in c.get("tags", [])]]


def resolve_email(name_or_group: str) -> list[str]:
    """Resolve a name or group to email addresses."""
    # Try group first
    group_contacts = lookup_group(name_or_group)
    if group_contacts:
        return [c["email"] for c in group_contacts if c.get("email")]
    # Try individual name
    matches = lookup(name_or_group, threshold=0.6)
    if matches:
        return [c["email"] for c in matches[:1] if c.get("email")]
    return []


def add_contact(
    name: str,
    email: str = "",
    phone: str = "",
    tags: list[str] | None = None,
    notes: str = "",
) -> dict:
    """Add a new contact."""
    contacts = _load()
    contact = {
        "id": _next_id(contacts),
        "name": name,
        "email": email,
        "phone": phone,
        "tags": tags or [],
        "notes": notes,
    }
    contacts.append(contact)
    _save(contacts)
    log.info("Contact added: #%d %s", contact["id"], name)
    return contact


def remove_contact(contact_id: int) -> bool:
    """Remove a contact by ID."""
    contacts = _load()
    before = len(contacts)
    contacts = [c for c in contacts if c.get("id") != contact_id]
    if len(contacts) < before:
        _save(contacts)
        return True
    return False


def list_all() -> list[dict]:
    """Return all contacts."""
    return _load()

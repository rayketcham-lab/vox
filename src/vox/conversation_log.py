"""Conversation logging — auto-logs all exchanges to JSONL files.

Local-only, privacy-first. One file per day in data/conversations/.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

_LOG_DIR: Path | None = None


def _get_log_dir() -> Path:
    global _LOG_DIR
    if _LOG_DIR is None:
        from vox.config import PROJECT_ROOT
        _LOG_DIR = PROJECT_ROOT / "data" / "conversations"
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR


def log_exchange(user_message: str, response: str, tools_used: list[str] | None = None):
    """Append a user/assistant exchange to today's conversation log."""
    try:
        log_dir = _get_log_dir()
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = log_dir / f"{today}.jsonl"

        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user_message,
            "assistant": response,
        }
        if tools_used:
            entry["tools"] = tools_used

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.debug("Conversation log failed: %s", e)


def get_recent(days: int = 1, limit: int = 20) -> list[dict]:
    """Retrieve recent conversation entries."""
    log_dir = _get_log_dir()
    entries = []

    # Collect log files from recent days
    from datetime import timedelta
    for d in range(days):
        date_str = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        log_file = log_dir / f"{date_str}.jsonl"
        if log_file.exists():
            for line in log_file.read_text(encoding="utf-8").strip().splitlines():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Most recent first, limited
    entries.reverse()
    return entries[:limit]


def search(keyword: str, days: int = 30) -> list[dict]:
    """Search past conversations by keyword."""
    keyword_lower = keyword.lower()
    results = []
    log_dir = _get_log_dir()

    from datetime import timedelta
    for d in range(days):
        date_str = (datetime.now() - timedelta(days=d)).strftime("%Y-%m-%d")
        log_file = log_dir / f"{date_str}.jsonl"
        if log_file.exists():
            for line in log_file.read_text(encoding="utf-8").strip().splitlines():
                try:
                    entry = json.loads(line)
                    if (keyword_lower in entry.get("user", "").lower()
                            or keyword_lower in entry.get("assistant", "").lower()):
                        results.append(entry)
                except json.JSONDecodeError:
                    continue

    return results[:20]

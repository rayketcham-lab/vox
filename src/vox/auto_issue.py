"""Auto-issue creation — detect unimplemented feature requests and file GitHub issues.

When a user asks for something VOX can't do, auto-create a GitHub issue to track it.
Deduplicates against existing open issues to avoid spam.
"""

from __future__ import annotations

import logging
import re
import subprocess

log = logging.getLogger(__name__)

# Patterns that suggest the user wants a feature VOX doesn't have
_FEATURE_REQUEST_SIGNALS = re.compile(
    r"\b(can\s+you|could\s+you|do\s+you|are\s+you\s+able\s+to|is\s+there\s+a\s+way\s+to)\b"
    r".*\b(play\s+music|set\s+a?\s*(timer|alarm|reminder)|read\s+(my\s+)?(clipboard|notifications)"
    r"|control\s+(my\s+)?(lights?|thermostat|tv|speaker|fan|door|lock)"
    r"|run\s+(a\s+)?(script|code|python|command)"
    r"|schedule|remind\s+me|wake\s+me|call\s+me"
    r"|translate|transcribe|summarize\s+(this\s+)?(file|document|page|video)"
    r"|upscale|enhance\s+(this\s+)?(image|photo|picture)"
    r"|clone\s+(my\s+)?voice|sound\s+like|speak\s+like"
    r"|connect\s+to|integrate\s+with|sync\s+with"
    r"|backup|export|import|convert)\b",
    re.IGNORECASE,
)

# Known capabilities — don't create issues for things that already work
_KNOWN_CAPABILITIES = {
    "weather", "time", "date", "search", "email", "selfie", "image",
    "picture", "photo", "map", "satellite", "fetch", "download", "pdf",
    "system", "gpu", "vram",
}

# Issue cache — avoid duplicate API calls within a session
_recent_issues: set[str] = set()


def _normalize_feature(text: str) -> str:
    """Extract a short feature description from user text."""
    # Strip command prefixes
    cleaned = re.sub(
        r"^(can you|could you|please|hey vox|vox|is there a way to|do you)\s+",
        "", text.strip(), flags=re.IGNORECASE,
    )
    # Truncate to something reasonable for an issue title
    cleaned = cleaned[:100].strip().rstrip("?.!")
    return cleaned


def should_create_issue(user_text: str, tool_result: str | None = None) -> bool:
    """Check if this request represents an unimplemented feature.

    Returns True if:
    - User is asking for something that matches feature request patterns
    - AND it's not a known capability
    - AND we haven't already filed it this session
    """
    from vox.config import GITHUB_REPO
    if not GITHUB_REPO:
        return False

    text_lower = user_text.lower()

    # Don't file issues for things we already do
    if any(cap in text_lower for cap in _KNOWN_CAPABILITIES):
        return False

    # Check if it matches a feature request pattern
    if not _FEATURE_REQUEST_SIGNALS.search(user_text):
        # Also check if a tool returned "Unknown tool" or similar failure
        if tool_result and ("unknown tool" in tool_result.lower() or "not configured" in tool_result.lower()):
            pass  # Allow issue creation for tool failures
        else:
            return False

    # Dedupe within session
    feature_key = _normalize_feature(user_text)[:50].lower()
    if feature_key in _recent_issues:
        return False

    return True


def create_feature_issue(user_text: str, context: str = "") -> str | None:
    """Create a GitHub issue for an unimplemented feature request.

    Uses `gh` CLI (already installed for the project).
    Returns the issue URL, or None if creation fails.
    """
    from vox.config import GITHUB_REPO
    if not GITHUB_REPO:
        return None

    feature = _normalize_feature(user_text)
    title = f"Feature request: {feature}"

    # Check for duplicate open issues first
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", GITHUB_REPO,
             "--state", "open", "--limit", "50", "--json", "title"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            existing = json.loads(result.stdout)
            existing_titles = {i["title"].lower() for i in existing}
            # Fuzzy match — check if key words overlap
            feature_words = set(feature.lower().split())
            for existing_title in existing_titles:
                existing_words = set(existing_title.lower().split())
                overlap = feature_words & existing_words
                if len(overlap) >= 3:  # 3+ words in common = likely duplicate
                    log.info("Skipping duplicate issue: %s (matches: %s)", title, existing_title)
                    _recent_issues.add(feature[:50].lower())
                    return None
    except Exception:
        log.debug("Could not check existing issues — proceeding with creation")

    body = (
        f"## User Request\n"
        f"> {user_text}\n\n"
        f"## Context\n"
        f"{context or 'Auto-detected as unimplemented feature.'}\n\n"
        f"## Notes\n"
        f"- Auto-created by VOX when user requested a feature that doesn't exist yet\n"
        f"- Review and prioritize accordingly\n"
    )

    try:
        result = subprocess.run(
            ["gh", "issue", "create", "--repo", GITHUB_REPO,
             "--title", title, "--body", body,
             "--label", "enhancement"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            issue_url = result.stdout.strip()
            log.info("Created GitHub issue: %s", issue_url)
            _recent_issues.add(feature[:50].lower())
            return issue_url
        else:
            log.warning("gh issue create failed: %s", result.stderr)
            return None
    except FileNotFoundError:
        log.warning("gh CLI not found — install GitHub CLI for auto-issue creation")
        return None
    except Exception as e:
        log.exception("Failed to create GitHub issue: %s", e)
        return None

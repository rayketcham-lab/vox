"""Claude API escalation — smart fallback for complex tasks.

When the local LLM (dolphin-mistral, etc.) can't handle a request well —
coding questions, complex reasoning, factual accuracy — escalate to Claude
Haiku for a cheap, high-quality answer.

Uses Haiku by default (~$0.001 per request). The local LLM stays primary
for personality, tool calling, and casual chat.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Patterns that suggest the local LLM might struggle
_ESCALATION_TRIGGERS = re.compile(
    r"\b(explain|how\s+does|how\s+do\s+you|how\s+to|why\s+does|why\s+is|what\s+is\s+the\s+difference)"
    r".*\b(code|program|algorithm|function|class|api|database|sql|python|javascript|"
    r"react|docker|kubernetes|linux|git|regex|http|tcp|ssl|encryption|hash|"
    r"machine\s+learning|neural|transformer|quantum|calculus|physics|chemistry|biology)\b"
    r"|\b(write|code|implement|create|build)\s+(me\s+)?(a\s+)?"
    r"(script|function|class|program|app|api|query|regex|dockerfile|makefile|config)\b"
    r"|\b(debug|fix|troubleshoot)\s+(this|my|the|a)?\s*"
    r"(code|script|function|error|bug|issue|problem|crash)\b"
    r"|\b(translate|convert|parse|compile|optimize|refactor)\s+(this|the|my)?\s*"
    r"(code|script|function|query|config|json|yaml|xml|csv)\b"
    r"|\b(summarize|analyze|compare|contrast|evaluate|critique)\s+"
    r"(this|the|these|those)?\s*(article|paper|document|report|text|data)\b",
    re.IGNORECASE,
)

# Phrases that indicate the local LLM gave a bad answer
_QUALITY_COMPLAINTS = re.compile(
    r"\b(that'?s?\s+(wrong|incorrect|not\s+right|inaccurate|bad|terrible|awful))"
    r"|\b(you'?re\s+(wrong|incorrect|confused|making\s+stuff\s+up|hallucinating))"
    r"|\b(no\s+that'?s\s+not|actually\s+it'?s|that\s+doesn'?t\s+(make\s+sense|work|sound\s+right))"
    r"|\b(try\s+again|think\s+harder|be\s+more\s+accurate|give\s+me\s+a\s+real\s+answer)\b",
    re.IGNORECASE,
)


def should_escalate(user_text: str, llm_response: str | None = None) -> bool:
    """Check if this request should be escalated to Claude.

    Returns True if:
    - The request matches complex task patterns, OR
    - The user is complaining about answer quality
    """
    from vox.config import CLAUDE_API_KEY
    if not CLAUDE_API_KEY:
        return False

    # User is unhappy with the local LLM's answer
    if _QUALITY_COMPLAINTS.search(user_text):
        return True

    # Complex task that local LLM likely can't handle well
    if _ESCALATION_TRIGGERS.search(user_text):
        return True

    return False


def escalate_to_claude(
    user_text: str,
    conversation_history: list[dict] | None = None,
    system_prompt: str = "",
) -> str | None:
    """Send a request to Claude API for a high-quality answer.

    Returns the response text, or None if escalation fails/unavailable.
    """
    from vox.config import CLAUDE_API_KEY, CLAUDE_MODEL

    if not CLAUDE_API_KEY:
        log.debug("Claude API key not set — escalation unavailable")
        return None

    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — pip install anthropic")
        return None

    try:
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

        # Build messages from conversation history
        messages = []
        if conversation_history:
            for entry in conversation_history[-10:]:  # last 10 messages
                role = entry.get("role", "")
                content = entry.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})

        # Always include the current user message
        if not messages or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})

        # Ensure messages alternate properly (Claude requirement)
        cleaned = []
        for msg in messages:
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += "\n" + msg["content"]
            else:
                cleaned.append(msg)
        # Must start with user
        if cleaned and cleaned[0]["role"] != "user":
            cleaned = cleaned[1:]

        if not cleaned:
            return None

        log.info("Escalating to Claude %s: %s", CLAUDE_MODEL, user_text[:80])

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=system_prompt or "You are a helpful assistant. Be concise — 1-3 sentences unless detail is needed.",
            messages=cleaned,
        )

        result = response.content[0].text
        log.info("Claude response (%d chars): %s", len(result), result[:200])
        return result

    except Exception as e:
        log.exception("Claude escalation failed: %s", e)
        return None

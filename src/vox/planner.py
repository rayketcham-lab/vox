"""Multi-step task planner — decomposes complex requests into tool chains.

Detects when a user request requires multiple sequential tool calls
with data flowing between them, generates a plan, and executes it.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Patterns that suggest multi-step requests (conjunctions + action words)
_MULTI_STEP_SIGNALS = re.compile(
    r"\b(then|and\s+then|after\s+that|next|finally|also|afterwards)\b"
    r".*\b(search|find|look|email|send|generate|create|save|download|fetch)\b",
    re.IGNORECASE,
)

_COMPLEX_REQUEST = re.compile(
    r"\b(research|compare|summarize|analyze|compile|gather)\b"
    r".*\b(and|then)\b"
    r".*\b(email|send|save|report|summary)\b",
    re.IGNORECASE,
)


def is_multi_step(text: str) -> bool:
    """Detect if a request likely needs multi-step planning."""
    if _MULTI_STEP_SIGNALS.search(text):
        return True
    if _COMPLEX_REQUEST.search(text):
        return True
    return False


def build_plan_prompt(user_request: str) -> str:
    """Build a prompt that asks the LLM to generate an execution plan."""
    return f"""The user wants a multi-step task done. Break it into ordered steps.
Each step should use one of these tools: web_search, web_fetch, send_email, \
generate_image, get_weather, add_note, lookup_contact.

User request: "{user_request}"

Respond with a numbered plan like:
1. [tool_name] — description of what to do
2. [tool_name] — description, using result from step 1
3. [tool_name] — description

Keep it to 2-5 steps. Be specific about what data flows between steps."""


def format_plan_for_user(plan_text: str) -> str:
    """Format the plan into a readable message for the user."""
    lines = [line.strip() for line in plan_text.strip().split("\n") if line.strip()]
    if not lines:
        return "I'll handle that for you."
    steps = [line for line in lines if re.match(r"^\d+\.", line)]
    if not steps:
        return plan_text
    return "Here's my plan:\n" + "\n".join(steps)

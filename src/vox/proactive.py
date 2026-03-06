"""Proactive behaviors — makes the persona feel alive.

Generates unprompted messages at natural intervals:
- Morning briefing (weather + reminders + a personal comment)
- Random check-ins ("hey, whatcha up to?")
- Conversation starters based on time-of-day activities
- Goodnight messages

All messages are generated through the LLM so they sound natural
and stay in-character. Frequency is randomized to avoid feeling robotic.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime

log = logging.getLogger(__name__)

# Track what we've already sent today to avoid repeats
_sent_today: set[str] = set()
_last_checkin_hour: int = -1


def reset_daily() -> None:
    """Reset daily tracking (call at midnight or on new day)."""
    global _last_checkin_hour
    _sent_today.clear()
    _last_checkin_hour = -1


def get_proactive_message() -> str | None:
    """Check if it's time for a proactive message. Returns a prompt to send
    to the LLM, or None if nothing is due.

    Called every ~60 seconds from the background loop. Uses randomization
    so messages don't fire at exact times.
    """
    global _last_checkin_hour

    now = datetime.now()
    hour = now.hour
    day = now.day

    # Reset tracking on new day
    if "day" not in _sent_today or int(list(_sent_today - {"day"})[0].split("_")[-1]) != day if any("day_" in s for s in _sent_today) else True:
        if f"day_{day}" not in _sent_today:
            reset_daily()
            _sent_today.add(f"day_{day}")

    # Morning briefing (7-9 AM, once per day)
    if 7 <= hour <= 9 and "morning" not in _sent_today:
        # 30% chance each check (so it feels natural, not exactly at 7:00)
        if random.random() < 0.3:
            _sent_today.add("morning")
            return _morning_briefing_prompt()

    # Random check-in (10 AM - 9 PM, max once every 2 hours)
    if 10 <= hour <= 21 and (hour - _last_checkin_hour) >= 2:
        # 5% chance each minute = roughly once per 20 minutes on average
        # but capped at once per 2 hours
        if random.random() < 0.05:
            _last_checkin_hour = hour
            _sent_today.add(f"checkin_{hour}")
            return _checkin_prompt()

    # Goodnight (10-11 PM, once per day)
    if 22 <= hour <= 23 and "goodnight" not in _sent_today:
        if random.random() < 0.2:
            _sent_today.add("goodnight")
            return _goodnight_prompt()

    return None


def _morning_briefing_prompt() -> str:
    """Generate prompt for morning briefing."""
    now = datetime.now()
    day_name = now.strftime("%A")
    date_str = now.strftime("%B %d")

    return (
        f"[PROACTIVE — you're initiating conversation, not responding to the user]\n"
        f"It's {day_name} morning, {date_str}. Say good morning in your own style. "
        f"Be brief and natural — like texting a friend. Maybe mention what you're "
        f"up to this morning (based on your activities). Keep it to 1-2 sentences. "
        f"Don't ask 'how can I help you' — you're a friend, not a service desk."
    )


def _checkin_prompt() -> str:
    """Generate prompt for a random check-in."""
    now = datetime.now()
    period = "morning" if now.hour < 12 else "afternoon" if now.hour < 17 else "evening"

    prompts = [
        f"[PROACTIVE — you're initiating conversation, not responding to the user]\n"
        f"It's {period}. Send a casual check-in — like a friend texting. "
        f"Maybe share something you're doing, a random thought, ask what they're up to, "
        f"or react to something from your activities/opinions. 1 sentence max. "
        f"Be natural, not formulaic.",

        f"[PROACTIVE — you're initiating conversation, not responding to the user]\n"
        f"Share a random thought or observation. Maybe something about "
        f"what you're watching, eating, thinking about. Keep it casual and short — "
        f"like a text from a friend. 1 sentence.",

        f"[PROACTIVE — you're initiating conversation, not responding to the user]\n"
        f"Send a short message — could be a question, a comment about your day, "
        f"a recommendation, or just vibes. Whatever feels natural right now. "
        f"1 sentence, casual tone.",
    ]
    return random.choice(prompts)


def _goodnight_prompt() -> str:
    """Generate prompt for goodnight message."""
    return (
        "[PROACTIVE — you're initiating conversation, not responding to the user]\n"
        "It's getting late. Say goodnight in your own way — casual, brief, maybe a little "
        "sweet. 1 sentence max. Don't be formal about it."
    )

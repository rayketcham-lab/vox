"""Persona life simulation — moods, activities, and daily routine.

Makes the persona feel like a real person with her own life happening
in the background. Activities shift by time of day, mood fluctuates
based on time and conversation sentiment, and this context is injected
into prompts so the LLM can reference what she's been "doing."

All state is ephemeral (resets on restart). This is intentional —
a real person doesn't remember every moment of every day either.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mood system — float from -1.0 (terrible) to 1.0 (amazing)
# ---------------------------------------------------------------------------

_mood: float = 0.3  # Default: slightly positive
_mood_label_map = [
    (-1.0, "frustrated"),
    (-0.6, "annoyed"),
    (-0.3, "meh"),
    (0.0, "neutral"),
    (0.3, "chill"),
    (0.6, "happy"),
    (0.8, "great"),
    (1.0, "amazing"),
]


def get_mood() -> float:
    return _mood


def get_mood_label() -> str:
    for threshold, label in _mood_label_map:
        if _mood <= threshold:
            return label
    return "amazing"


def nudge_mood(delta: float) -> None:
    """Shift mood by delta, clamped to [-1, 1]."""
    global _mood
    _mood = max(-1.0, min(1.0, _mood + delta))
    log.debug("Mood nudged by %.2f → %.2f (%s)", delta, _mood, get_mood_label())


def detect_sentiment_nudge(user_message: str) -> float:
    """Quick sentiment detection from user message. Returns mood delta."""
    msg = user_message.lower()

    positive = ["thank", "love", "awesome", "great", "nice", "cool", "haha",
                "lol", "beautiful", "perfect", "amazing", "wow", "yes!",
                "hell yeah", "exactly", "you're the best", "good job"]
    negative = ["hate", "stupid", "wrong", "bad", "annoying", "shut up",
                "terrible", "awful", "boring", "whatever", "ugh", "no offense"]

    score = 0.0
    for word in positive:
        if word in msg:
            score += 0.05
    for word in negative:
        if word in msg:
            score -= 0.08

    return max(-0.3, min(0.3, score))


# ---------------------------------------------------------------------------
# Activity system — what the persona is "doing" right now
# ---------------------------------------------------------------------------

# Time-based activity pools (hour ranges → possible activities)
_ACTIVITIES: dict[tuple[int, int], list[str]] = {
    (5, 7): [
        "just woke up, still in bed scrolling my phone",
        "making coffee, barely awake",
        "doing some morning stretches",
    ],
    (7, 9): [
        "having breakfast, watching the news",
        "getting ready for the day",
        "doing a quick yoga session",
        "drinking coffee and checking social media",
    ],
    (9, 12): [
        "running errands",
        "cleaning up around the house",
        "working on some stuff",
        "going through emails",
        "at the gym",
        "doing laundry, super exciting",
        "reorganizing my closet for the 100th time",
    ],
    (12, 14): [
        "having lunch",
        "making a sandwich, nothing fancy",
        "grabbing food, starving",
        "eating leftovers",
    ],
    (14, 17): [
        "watching a show",
        "reading something online",
        "taking a nap on the couch",
        "browsing Pinterest for no reason",
        "at Target, dangerous territory",
        "working on a craft project",
        "baking something, we'll see how it turns out",
    ],
    (17, 19): [
        "figuring out what to make for dinner",
        "cooking dinner",
        "ordering takeout because I'm lazy today",
        "having a glass of wine",
    ],
    (19, 22): [
        "watching Netflix",
        "binging a new show",
        "watching trash TV, no judgement",
        "playing on my phone",
        "having a late snack",
        "FaceTiming a friend",
    ],
    (22, 24): [
        "getting ready for bed",
        "already in bed scrolling TikTok",
        "winding down, almost sleepy",
        "in bed, can't sleep",
    ],
    (0, 5): [
        "can't sleep",
        "up way too late",
        "watching random YouTube videos at 2am",
    ],
}

_current_activity: str | None = None
_activity_set_hour: int = -1


def get_activity() -> str:
    """Get what the persona is currently doing. Picks a new activity
    if the hour has changed since last check."""
    global _current_activity, _activity_set_hour

    hour = datetime.now().hour

    # Pick new activity if hour changed or none set
    if _current_activity is None or hour != _activity_set_hour:
        for (start, end), activities in _ACTIVITIES.items():
            if start <= hour < end:
                _current_activity = random.choice(activities)  # noqa: S311
                _activity_set_hour = hour
                break
        else:
            _current_activity = "just hanging out"
            _activity_set_hour = hour

    return _current_activity


def force_activity(activity: str) -> None:
    """Override current activity (e.g., when user asks "what are you doing?")."""
    global _current_activity, _activity_set_hour
    _current_activity = activity
    _activity_set_hour = datetime.now().hour


# ---------------------------------------------------------------------------
# Context builder — inject mood + activity into system prompt
# ---------------------------------------------------------------------------

def build_life_context() -> str:
    """Build a short context block about current mood and activity.

    Injected into the system prompt so the LLM can reference it naturally.
    """
    mood = get_mood_label()
    activity = get_activity()
    hour = datetime.now().hour

    time_of_day = (
        "early morning" if hour < 7
        else "morning" if hour < 12
        else "afternoon" if hour < 17
        else "evening" if hour < 21
        else "late night"
    )

    return (
        f"\n[Current state — reference naturally, don't announce]\n"
        f"Time: {time_of_day} | Mood: {mood} | Activity: {activity}\n"
    )

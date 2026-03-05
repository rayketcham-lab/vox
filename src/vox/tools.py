"""Tool definitions and intent detection for concurrent execution."""

from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent detection — fast keyword matching, no LLM needed (~1ms)
# ---------------------------------------------------------------------------

@dataclass
class DetectedIntent:
    """A detected tool intent with the function to call."""

    tool_name: str
    args: dict
    bridge_phrase: str  # what VOX says while the tool runs


# Pattern → (tool_name, arg_builder, bridge_phrase)
# arg_builder receives (match, full_text) so it can extract args from the full user message
_INTENT_PATTERNS: list[tuple[re.Pattern, str, callable, str]] = []


def _add_pattern(pattern: str, tool_name: str, arg_builder: callable, bridge: str):
    _INTENT_PATTERNS.append((re.compile(pattern, re.IGNORECASE), tool_name, arg_builder, bridge))


def _extract_email(text: str) -> str:
    """Extract an email address from text, falling back to USER_EMAIL config."""
    m = re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text)
    if m:
        return m.group(0)
    from vox.config import USER_EMAIL
    return USER_EMAIL


def _extract_url(text: str) -> str:
    """Extract a URL from text."""
    m = re.search(r"https?://\S+", text)
    return m.group(0).rstrip(".,;!?)") if m else ""


def _is_selfie_request(text: str) -> bool:
    """Check if the text is asking for a selfie / picture of the assistant."""
    return bool(re.search(
        r"\b(selfie|selfy)\b"
        r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself|your\s*(self|face|body))"
        r"|\b(what\s+do\s+you\s+look\s+like)"
        r"|\b(show\s+(me\s+)?(yourself|what\s+you\s+look\s+like))"
        r"|\b(take\s+a\s+(pic|picture|photo|selfie|snap|shot))\b"
        r"|\b(send\s+me\s+a\s+(selfie|pic|picture|photo)\s+(of\s+)?(you|yourself))"
        r"|\b(let\s+me\s+see\s+you)\b",
        text, re.IGNORECASE,
    ))


def _build_persona_prompt(text: str) -> str:
    """Build an SD prompt using the persona description + any scene context from the user."""
    from vox.config import VOX_PERSONA_DESCRIPTION, VOX_PERSONA_STYLE

    # Extract scene/context modifiers from the user's request
    scene = text.strip()
    # Strip common prefixes
    scene = re.sub(r"^(can you|could you|please|hey vox|vox)\s+", "", scene, flags=re.IGNORECASE)
    # Strip selfie-related command words
    scene = re.sub(
        r"^(send|give|show|take|email|mail)\s+(me\s+)?(a\s+)?",
        "", scene, flags=re.IGNORECASE,
    )
    scene = re.sub(
        r"^(selfie|selfy|pic|picture|photo|image|snap|shot)\s*(of\s+)?(you|yourself)?\s*",
        "", scene, flags=re.IGNORECASE,
    )
    # Strip "what do you look like" style queries
    scene = re.sub(r"^(what\s+do\s+you\s+look\s+like)\s*", "", scene, flags=re.IGNORECASE)
    scene = re.sub(r"^(show\s+(me\s+)?(yourself|what\s+you\s+look\s+like))\s*", "", scene, flags=re.IGNORECASE)
    scene = re.sub(r"^(let\s+me\s+see\s+you)\s*", "", scene, flags=re.IGNORECASE)
    # Strip email-related tail
    scene = re.sub(r"\b(and\s+)?(email|send|mail)\b.*$", "", scene, flags=re.IGNORECASE).strip()
    scene = re.sub(r"\b(at|to)\s+\S+@\S+\.\S+.*$", "", scene, flags=re.IGNORECASE).strip()
    # Clean up remaining artifacts
    scene = re.sub(r"^(at\s+the|at|in\s+the|in|on\s+the|on|by\s+the|by|with)\s+", r"\g<0>", scene, flags=re.IGNORECASE)
    scene = scene.strip().rstrip("?.!,")

    # Build the full prompt: persona description + scene context + style
    parts = []
    if VOX_PERSONA_DESCRIPTION:
        parts.append(VOX_PERSONA_DESCRIPTION)
    if scene:
        parts.append(scene)
    if VOX_PERSONA_STYLE:
        parts.append(VOX_PERSONA_STYLE)

    return ", ".join(parts) if parts else "a portrait"


def _extract_image_prompt(text: str) -> str:
    """Extract an image prompt by stripping command words and filler."""
    prompt = re.sub(
        r"^(can you|could you|please|hey vox|vox)\s+",
        "", text.strip(), flags=re.IGNORECASE,
    )
    # Strip "email/mail/send/give/show me" prefix
    prompt = re.sub(
        r"^(email|mail|send|give|show)\s+me\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip count prefix ("5 pictures of" → "")
    prompt = re.sub(
        r"^\d+\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip command verbs
    prompt = re.sub(
        r"^(generate|create|draw|make|paint|imagine)\s+",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip "me" after command verb
    prompt = re.sub(r"^me\s+", "", prompt, flags=re.IGNORECASE)
    # Strip "an image/picture/photo/pics of"
    prompt = re.sub(
        r"^(an?\s+)?(image|picture|photo|artwork|illustration|pic|pics|pictures|photos|images)\s+(of\s+|with\s+)?",
        "", prompt, flags=re.IGNORECASE,
    )
    # Strip email-related tail ("...and email it to foo@bar.com", "...at foo@bar.com")
    prompt = re.sub(r"\b(and\s+)?(email|send)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r"\b(at|to)\s+\S+@\S+\.\S+.*$", "", prompt, flags=re.IGNORECASE).strip()
    # Strip purpose tails — "for a ... meme", "for my blog", etc.
    prompt = re.sub(r"\s+for\s+(a|an|my|the|some)\b.*\b(meme|blog|post|project|website|collection)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    # Strip conversational tails — "but we should...", "and we need...", "we should...", etc.
    prompt = re.sub(r",?\s*\b(but|however)\s+(we|you|i|it)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r",?\s*\b(and|,)\s+(we|you|i)\s+(should|need|want|can|could|have)\b.*$", "", prompt, flags=re.IGNORECASE).strip()
    return prompt.strip().rstrip("?.!")


def _build_search_query(text: str) -> str:
    """Build a search query by stripping command words and email addresses."""
    # Remove common command prefixes
    q = re.sub(
        r"^(can you|could you|please|hey vox|vox)\s+",
        "", text.strip(), flags=re.IGNORECASE,
    )
    # Remove email-related tail ("...email me at foo@bar.com...")
    q = re.sub(r"\b(and\s+)?(can you\s+)?email\b.*$", "", q, flags=re.IGNORECASE).strip()
    # Remove trailing punctuation
    q = q.rstrip("?.!")
    return q


# Register intent patterns
_add_pattern(
    r"weather|forecast|temperature|rain|sunny|snow",
    "get_weather",
    lambda m, t: {},
    "Let me check the forecast for you...",
)
_add_pattern(
    r"what time|current time|what.s the time",
    "get_current_time",
    lambda m, t: {},
    "The time right now is",
)
_add_pattern(
    r"system info|gpu|vram|memory|cpu info",
    "get_system_info",
    lambda m, t: {},
    "Let me check the system...",
)
_add_pattern(
    r"\b(search|look\s*up|find|google|search\s+for)\b",
    "web_search",
    lambda m, t: {"query": _build_search_query(t)},
    "Let me search for that...",
)
_add_pattern(
    r"\b(download|fetch|open|get|grab)\b.*\b(pdf|page|url|link|site|website)\b",
    "web_fetch",
    lambda m, t: {"url": _extract_url(t)},
    "Let me fetch that for you...",
)
_add_pattern(
    r"https?://\S+",
    "web_fetch",
    lambda m, t: {"url": _extract_url(t)},
    "Let me fetch that for you...",
)
# Email with explicit address (highest priority for email)
_add_pattern(
    r"\b(email|mail)\b.*\b\S+@\S+\.\S+",
    "send_email",
    lambda m, t: {"to": _extract_email(t)},
    "I'll send that over...",
)
# Image generation patterns — BEFORE generic "email/mail me" so
# "email me a picture" routes to generate_image (not send_email)
_add_pattern(
    r"\b(generate|create|draw|make|paint|imagine)\b.*\b(image|picture|photo|artwork|illustration)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
_add_pattern(
    r"\b(draw|paint)\s+me\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
_add_pattern(
    r"\bimagine\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "email/mail/send/give me a picture of X" — implies generation
_add_pattern(
    r"\b(email|mail|send|give|show)\s+me\b.*\b(image|picture|photo|pic|pics)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "give me N pictures of X" / "show me X" — broader image triggers
_add_pattern(
    r"\b(give|show|get)\s+me\b.*\b(picture|image|photo|pic|pics)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# "\d+ pictures/images of" — count-based trigger
_add_pattern(
    r"\b\d+\s+(picture|image|photo|pic|pics)\w*\s+(of|with)\b",
    "generate_image",
    lambda m, t: {"prompt": _extract_image_prompt(t)},
    "Let me generate that image for you...",
)
# Selfie / persona-aware image triggers — "send me a selfie", "take a pic", "what do you look like"
_add_pattern(
    r"\b(selfie|selfy)\b"
    r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself)"
    r"|\bwhat\s+do\s+you\s+look\s+like\b"
    r"|\bshow\s+(me\s+)?(yourself|what\s+you\s+look\s+like)"
    r"|\btake\s+a\s+(pic|picture|photo|selfie|snap|shot)\b"
    r"|\bsend\s+me\s+a\s+(selfie|pic|picture|photo)\s+(of\s+)?(you|yourself)"
    r"|\blet\s+me\s+see\s+you\b",
    "generate_image",
    lambda m, t: {"prompt": _build_persona_prompt(t), "_selfie": True},
    "Let me take a pic for you...",
)
# Generic "email/mail me" / "send me" (no address) — AFTER image patterns
_add_pattern(
    r"\b(email|mail|send)\s+(me|it|this|that|the)\b",
    "send_email",
    lambda m, t: {"to": _extract_email(t)},
    "I'll send that over...",
)


def detect_intent(text: str) -> DetectedIntent | None:
    """Fast intent detection via regex. Returns first match or None."""
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        match = pattern.search(text)
        if match:
            intent = DetectedIntent(
                tool_name=tool_name,
                args=arg_builder(match, text),
                bridge_phrase=bridge,
            )
            log.info("Intent detected: %s args=%s", tool_name, intent.args)
            return intent
    log.debug("No intent detected for: %s", text[:80])
    return None


def detect_all_intents(text: str) -> list[DetectedIntent]:
    """Detect ALL matching intents in the text (for tool chaining)."""
    intents = []
    seen = set()
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        if tool_name in seen:
            continue
        match = pattern.search(text)
        if match:
            intents.append(DetectedIntent(
                tool_name=tool_name,
                args=arg_builder(match, text),
                bridge_phrase=bridge,
            ))
            seen.add(tool_name)
    if intents:
        log.info("All intents: %s", [i.tool_name for i in intents])
    return intents


# Validation patterns — stricter than intent detection.
# Used to block spurious LLM-initiated tool calls that don't match the user's actual request.
_TOOL_VALIDATORS: dict[str, re.Pattern] = {
    "get_weather": re.compile(
        r"\b(weather|forecast|temperature|rain(?:ing)?|sunny|snow(?:ing)?|storm|humid|wind|cold|hot|warm|cool)\b",
        re.IGNORECASE,
    ),
    "get_current_time": re.compile(
        r"\b(what time|current time|the time|the date|what day|today.s date)\b",
        re.IGNORECASE,
    ),
    "get_system_info": re.compile(
        r"\b(system info|gpu|vram|memory usage|cpu info|system stats|hardware)\b",
        re.IGNORECASE,
    ),
    "web_search": re.compile(
        r"\b(search|look\s*up|find|google|lookup)\b",
        re.IGNORECASE,
    ),
    "web_fetch": re.compile(
        r"\b(download|fetch|open|get|grab|pdf|page|url|link|site|website)\b|https?://\S+",
        re.IGNORECASE,
    ),
    "send_email": re.compile(
        r"\b(email|send|mail)\b.*(\S+@\S+\.\S+|\b(me|it|this|that|the|results|report)\b)",
        re.IGNORECASE,
    ),
    "generate_image": re.compile(
        r"\b(generate|create|draw|make|paint|imagine)\b.*\b(image|picture|photo|artwork|illustration)\b"
        r"|\b(draw|paint)\s+me\b"
        r"|\bimagine\b"
        r"|\b(email|mail|send|give|show)\s+me\b.*\b(image|picture|photo|pic|pics)\b"
        r"|\b\d+\s+(picture|image|photo|pic|pics)\w*\s+(of|with)\b"
        r"|\b(selfie|selfy)\b"
        r"|\b(picture|pic|photo|image)\s+(of\s+)?(you|yourself)"
        r"|\bwhat\s+do\s+you\s+look\s+like\b"
        r"|\btake\s+a\s+(pic|picture|photo|selfie|snap|shot)\b"
        r"|\blet\s+me\s+see\s+you\b",
        re.IGNORECASE,
    ),
}


def validate_tool_call(tool_name: str, user_message: str) -> bool:
    """Check if a tool call is actually relevant to the user's current message.

    This catches cases where the LLM hallucinates tool calls based on
    conversation history rather than the current request.
    """
    validator = _TOOL_VALIDATORS.get(tool_name)
    if validator is None:
        # Unknown tool — let it through (execute_tool will handle the error)
        return True
    return bool(validator.search(user_message))


# ---------------------------------------------------------------------------
# Tool definitions (Ollama format) — used as fallback when intent not detected
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_system_info",
            "description": "Get system information (GPU, memory, CPU)",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name (defaults to auto-detect)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Use when the user asks to find something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a URL and return its content. For HTML pages, returns extracted text. For PDFs, downloads and saves the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt using Stable Diffusion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "style": {
                        "type": "string",
                        "description": "Optional style modifier (e.g. 'photorealistic', 'watercolor', 'oil painting')",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient. Use when the user asks to email something to an address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content",
                    },
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths to attach",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_TOOL_REGISTRY: dict[str, callable] = {}


def _register(name: str):
    def decorator(fn):
        _TOOL_REGISTRY[name] = fn
        return fn

    return decorator


@_register("get_current_time")
def _get_current_time(**kwargs) -> str:
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")


@_register("get_system_info")
def _get_system_info(**kwargs) -> str:
    import platform

    lines = [
        f"OS: {platform.system()} {platform.release()}",
        f"Python: {platform.python_version()}",
    ]
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            lines.append(f"GPU: {gpu} ({vram:.1f} GB VRAM)")
    except ImportError:
        lines.append("GPU: torch not available")
    return "\n".join(lines)


@_register("get_weather")
def _get_weather(**kwargs) -> str:
    """Fetch weather from Open-Meteo API (free, no key needed)."""
    import json
    import urllib.request

    try:
        # Step 1: Get location from IP (free, no key)
        geo_url = "https://ipapi.co/json/"
        with urllib.request.urlopen(geo_url, timeout=3) as resp:
            geo = json.loads(resp.read())
        lat, lon = geo.get("latitude", 40.71), geo.get("longitude", -74.01)
        city = geo.get("city", "Unknown")

        # Step 2: Get forecast from Open-Meteo (free, no key)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max,weathercode"
            f"&temperature_unit=fahrenheit"
            f"&timezone=auto"
            f"&forecast_days=7"
        )
        with urllib.request.urlopen(weather_url, timeout=5) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        rain = daily.get("precipitation_probability_max", [])
        codes = daily.get("weathercode", [])

        code_map = {
            0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Rain showers", 82: "Heavy rain showers",
            95: "Thunderstorm",
        }

        lines = [f"7-day forecast for {city}:"]
        for i in range(min(7, len(dates))):
            condition = code_map.get(codes[i], f"Code {codes[i]}")
            lines.append(
                f"  {dates[i]}: {condition}, High {highs[i]:.0f}F / Low {lows[i]:.0f}F, "
                f"{rain[i]}% chance of rain"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Weather lookup failed: {e}"


def _clean_ddg_url(url: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect wrapper.

    DDG HTML lite wraps links like:
        //duckduckgo.com/l/?uddg=https%3A%2F%2Freal-url.com&rut=abc123
    This extracts and returns the decoded ``uddg`` parameter value.
    Non-redirect URLs are returned unchanged.
    """
    if not url:
        return url
    import urllib.parse as _urlparse

    parsed = _urlparse.urlparse(url)
    if parsed.hostname and "duckduckgo.com" in parsed.hostname and parsed.path.startswith("/l/"):
        qs = _urlparse.parse_qs(parsed.query)
        uddg = qs.get("uddg")
        if uddg:
            return uddg[0]
    return url


@_register("web_search")
def _web_search(query: str = "", **kwargs) -> str:
    """Search the web using DuckDuckGo HTML (no API key needed)."""
    import json
    import urllib.parse
    import urllib.request

    if not query:
        return "No search query provided."

    try:
        encoded = urllib.parse.urlencode({"q": query, "format": "json"})
        url = f"https://api.duckduckgo.com/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "VOX/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        results = []

        # Abstract (instant answer)
        if data.get("Abstract"):
            results.append(f"Summary: {data['Abstract']}")
            if data.get("AbstractURL"):
                results.append(f"Source: {data['AbstractURL']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                text = topic["Text"]
                url = topic.get("FirstURL", "")
                results.append(f"- {text}")
                if url:
                    results.append(f"  Link: {url}")

        if not results:
            # Fallback: try a scrape of DuckDuckGo HTML lite
            html_url = f"https://html.duckduckgo.com/html/?{urllib.parse.urlencode({'q': query})}"
            req = urllib.request.Request(html_url, headers={"User-Agent": "VOX/0.1"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Extract result snippets from HTML
            import re as _re

            snippets = _re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, _re.DOTALL)
            links = _re.findall(r'class="result__a"[^>]*href="([^"]*)"', html)
            if not links:
                links = _re.findall(r'class="result__url"[^>]*href="([^"]*)"', html)
            for i, snippet in enumerate(snippets[:5]):
                clean = _re.sub(r"<[^>]+>", "", snippet).strip()
                link = _clean_ddg_url(links[i]) if i < len(links) else ""
                results.append(f"- {clean}")
                if link:
                    results.append(f"  Link: {link}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n" + "\n".join(results)

    except Exception as e:
        return f"Search failed: {e}"


@_register("send_email")
def _send_email(
    to: str = "",
    subject: str = "",
    body: str = "",
    attachments: list[str] | str = "",
    **kwargs,
) -> str:
    """Send an email via SMTP, optionally with file attachments."""
    import mimetypes
    import os
    import smtplib
    from email import encoders
    from email.mime.base import MIMEBase
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from vox.config import SMTP_FROM, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER

    log.info("send_email called: to=%s, subject=%s, attachments=%s", to, subject, type(attachments).__name__)

    if not to:
        log.warning("send_email: no recipient provided")
        return "No recipient email address provided."
    if not SMTP_HOST:
        log.warning("send_email: SMTP_HOST not configured")
        return "Email not configured. Set SMTP_HOST in .env"

    log.info("SMTP config: host=%s, port=%s, from=%s, user=%s",
             SMTP_HOST, SMTP_PORT, SMTP_FROM, SMTP_USER or "(none)")

    # Normalize attachments to a list
    if isinstance(attachments, str):
        attachment_list = [attachments] if attachments else []
    else:
        attachment_list = list(attachments) if attachments else []

    warnings: list[str] = []

    try:
        if attachment_list:
            msg = MIMEMultipart()
            msg.attach(MIMEText(body))

            for filepath in attachment_list:
                if not os.path.isfile(filepath):
                    warnings.append(f"Attachment not found, skipped: {filepath}")
                    continue

                content_type, _ = mimetypes.guess_type(filepath)
                if content_type is None:
                    content_type = "application/octet-stream"
                maintype, subtype = content_type.split("/", 1)

                with open(filepath, "rb") as f:
                    part = MIMEBase(maintype, subtype)
                    part.set_payload(f.read())

                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    "attachment",
                    filename=os.path.basename(filepath),
                )
                msg.attach(part)
        else:
            msg = MIMEText(body)

        msg["Subject"] = subject or "Message from VOX"
        msg["From"] = SMTP_FROM or SMTP_USER or f"vox@{SMTP_HOST}"
        msg["To"] = to

        log.info("Connecting to SMTP %s:%s ...", SMTP_HOST, SMTP_PORT)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            has_tls = server.has_extn("starttls")
            log.info("STARTTLS available: %s", has_tls)
            if has_tls:
                server.starttls()
                server.ehlo()
            if SMTP_USER and SMTP_PASSWORD:
                log.info("Authenticating as %s ...", SMTP_USER)
                server.login(SMTP_USER, SMTP_PASSWORD)
            log.info("Sending message: from=%s to=%s subject=%s", msg["From"], to, msg["Subject"])
            server.send_message(msg)
            log.info("SMTP send_message completed successfully")

        result = f"Email sent to {to} with subject: {subject}"
        if warnings:
            result += "\nWarnings: " + "; ".join(warnings)
        log.info("send_email result: %s", result)
        return result

    except Exception as e:
        log.exception("send_email failed: %s", e)
        return f"Failed to send email: {e}"


@_register("web_fetch")
def _web_fetch(url: str = "", **kwargs) -> str:
    """Fetch a URL: return text for HTML, save file for PDFs."""
    import urllib.request

    from vox.config import DOWNLOADS_DIR

    if not url:
        return "No URL provided."
    if not re.match(r"https?://", url):
        return f"Invalid URL: {url}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VOX/0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            # Save PDF to downloads directory
            filename = url.split("/")[-1].split("?")[0]
            filename = re.sub(r"[^\w.\-]", "_", filename)
            if not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            filepath = DOWNLOADS_DIR / filename
            filepath.write_bytes(data)
            size_kb = len(data) / 1024
            return f"PDF saved to {filepath} ({size_kb:.1f} KB)"
        else:
            # HTML or other text — strip tags and return text
            text = data.decode("utf-8", errors="replace")
            text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 2000:
                text = text[:2000] + "..."
            return f"Content from {url}:\n{text}"

    except Exception as e:
        return f"Fetch failed: {e}"


@_register("generate_image")
def _generate_image(prompt: str = "", style: str = "", _selfie: bool = False, **kwargs) -> str:
    """Generate an image using Stable Diffusion."""
    from vox.config import (
        DOWNLOADS_DIR,
        IMAGE_HEIGHT,
        IMAGE_MODEL,
        IMAGE_NEGATIVE_PROMPT,
        IMAGE_NSFW_FILTER,
        IMAGE_STEPS,
        IMAGE_WIDTH,
        VOX_PERSONA_DESCRIPTION,
        VOX_PERSONA_STYLE,
    )

    if not prompt:
        return "No image prompt provided."

    # For selfie requests, the prompt is already built by _build_persona_prompt.
    # For non-selfie requests, check if the user is asking about the persona
    # and prepend persona description if configured.
    if _selfie:
        full_prompt = prompt  # already persona-aware from _build_persona_prompt
    elif style:
        full_prompt = f"{prompt}, {style}"
    else:
        full_prompt = prompt
    log.info("generate_image: prompt=%r, style=%r, model=%s", prompt, style, IMAGE_MODEL)
    log.info("generate_image: steps=%d, size=%dx%d, nsfw_filter=%s",
             IMAGE_STEPS, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_NSFW_FILTER)

    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        log.error("generate_image: diffusers not installed")
        return "Image generation requires the 'diffusers' package. Install with: pip install vox[image]"

    try:
        log.info("Loading Stable Diffusion pipeline: %s", IMAGE_MODEL)
        pipe_kwargs = {
            "torch_dtype": torch.float16,
        }
        if IMAGE_NSFW_FILTER.lower() == "off":
            log.info("NSFW safety checker disabled")
            pipe_kwargs["safety_checker"] = None

        pipe = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL, **pipe_kwargs)
        pipe = pipe.to("cuda")
        pipe.enable_attention_slicing()
        log.info("Pipeline loaded and moved to CUDA")

        log.info("Generating image: %r (negative: %r)", full_prompt, IMAGE_NEGATIVE_PROMPT[:60])
        result = pipe(
            full_prompt,
            negative_prompt=IMAGE_NEGATIVE_PROMPT or None,
            num_inference_steps=IMAGE_STEPS,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )
        image = result.images[0]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vox_image_{timestamp}.png"
        filepath = DOWNLOADS_DIR / filename
        image.save(filepath)
        log.info("Image saved to %s", filepath)

        return f"Image generated and saved to {filepath}"

    except Exception as e:
        log.exception("generate_image failed: %s", e)
        return f"Image generation failed: {e}"


def execute_tool(name: str, args: dict) -> str:
    """Execute a registered tool by name."""
    log.info("execute_tool: %s(%s)", name, args)
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        log.warning("Unknown tool: %s", name)
        return f"Unknown tool: {name}"
    try:
        result = fn(**args)
        log.info("execute_tool %s result (%d chars): %s", name, len(result), result[:200])
        return result
    except Exception as e:
        log.exception("Tool error in %s: %s", name, e)
        return f"Tool error: {e}"

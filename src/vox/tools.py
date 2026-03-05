"""Tool definitions and intent detection for concurrent execution."""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass

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
    """Extract an email address from text."""
    m = re.search(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", text)
    return m.group(0) if m else ""


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
    r"\bemail\b.*\b\S+@\S+\.\S+",
    "send_email",
    lambda m, t: {"to": _extract_email(t)},
    "I'll send that over...",
)


def detect_intent(text: str) -> DetectedIntent | None:
    """Fast intent detection via regex. Returns first match or None."""
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return DetectedIntent(
                tool_name=tool_name,
                args=arg_builder(match, text),
                bridge_phrase=bridge,
            )
    return None


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
    "send_email": re.compile(
        r"\b(email|send|mail)\b.*\S+@\S+\.\S+",
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
            links = _re.findall(r'class="result__url"[^>]*href="([^"]*)"', html)
            for i, snippet in enumerate(snippets[:5]):
                clean = _re.sub(r"<[^>]+>", "", snippet).strip()
                link = links[i] if i < len(links) else ""
                results.append(f"- {clean}")
                if link:
                    results.append(f"  Link: {link}")

        if not results:
            return f"No results found for: {query}"

        return f"Search results for '{query}':\n" + "\n".join(results)

    except Exception as e:
        return f"Search failed: {e}"


@_register("send_email")
def _send_email(to: str = "", subject: str = "", body: str = "", **kwargs) -> str:
    """Send an email via SMTP."""
    import smtplib
    from email.mime.text import MIMEText

    from vox.config import SMTP_FROM, SMTP_HOST, SMTP_PASSWORD, SMTP_PORT, SMTP_USER

    if not to:
        return "No recipient email address provided."
    if not SMTP_HOST:
        return "Email not configured. Set SMTP_HOST in .env"

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject or "Message from VOX"
        msg["From"] = SMTP_FROM or SMTP_USER or f"vox@{SMTP_HOST}"
        msg["To"] = to

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.ehlo()
            # Use STARTTLS if the server supports it
            if server.has_extn("starttls"):
                server.starttls()
                server.ehlo()
            # Authenticate if credentials are configured
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        return f"Email sent to {to} with subject: {subject}"

    except Exception as e:
        return f"Failed to send email: {e}"


def execute_tool(name: str, args: dict) -> str:
    """Execute a registered tool by name."""
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error: {e}"

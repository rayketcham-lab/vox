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
_INTENT_PATTERNS: list[tuple[re.Pattern, str, callable, str]] = []


def _add_pattern(pattern: str, tool_name: str, arg_builder: callable, bridge: str):
    _INTENT_PATTERNS.append((re.compile(pattern, re.IGNORECASE), tool_name, arg_builder, bridge))


# Register intent patterns
_add_pattern(
    r"weather|forecast|temperature|rain|sunny|snow",
    "get_weather",
    lambda m: {},
    "Let me check the forecast for you...",
)
_add_pattern(
    r"what time|current time|what.s the time",
    "get_current_time",
    lambda m: {},
    "The time right now is",
)
_add_pattern(
    r"system info|gpu|vram|memory|cpu info",
    "get_system_info",
    lambda m: {},
    "Let me check the system...",
)


def detect_intent(text: str) -> DetectedIntent | None:
    """Fast intent detection via regex. Returns first match or None."""
    for pattern, tool_name, arg_builder, bridge in _INTENT_PATTERNS:
        match = pattern.search(text)
        if match:
            return DetectedIntent(
                tool_name=tool_name,
                args=arg_builder(match),
                bridge_phrase=bridge,
            )
    return None


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
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
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


def execute_tool(name: str, args: dict) -> str:
    """Execute a registered tool by name."""
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error: {e}"

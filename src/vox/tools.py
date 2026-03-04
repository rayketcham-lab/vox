"""Tool definitions for LLM function calling."""

from __future__ import annotations

import datetime

# Tool definitions in Ollama format
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
]

# Tool implementations
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


def execute_tool(name: str, args: dict) -> str:
    """Execute a registered tool by name."""
    fn = _TOOL_REGISTRY.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return fn(**args)
    except Exception as e:
        return f"Tool error: {e}"

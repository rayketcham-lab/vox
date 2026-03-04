"""LLM interface using Ollama with tool calling support."""

from __future__ import annotations

import ollama

from vox.config import OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT
from vox.tools import TOOL_DEFINITIONS, execute_tool

# Conversation history for context
_history: list[dict] = []
MAX_HISTORY = 20


def _get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def chat(user_message: str, model_override: str | None = None) -> str:
    """Send a message to the LLM and return the response.

    Handles tool calling if the model requests it.
    """
    model = model_override or OLLAMA_MODEL
    client = _get_client()

    _history.append({"role": "user", "content": user_message})

    # Trim history if too long
    while len(_history) > MAX_HISTORY:
        _history.pop(0)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history]

    response = client.chat(
        model=model,
        messages=messages,
        tools=TOOL_DEFINITIONS if TOOL_DEFINITIONS else None,
    )

    msg = response["message"]

    # Handle tool calls
    if msg.get("tool_calls"):
        for tool_call in msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]
            print(f"[LLM] Calling tool: {fn_name}({fn_args})")
            result = execute_tool(fn_name, fn_args)
            _history.append({"role": "tool", "content": str(result)})

        # Get final response after tool execution
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history]
        response = client.chat(model=model, messages=messages)
        msg = response["message"]

    reply = msg.get("content", "")
    _history.append({"role": "assistant", "content": reply})
    print(f"[LLM] Response: {reply}")
    return reply


def clear_history() -> None:
    """Clear conversation history."""
    _history.clear()

"""LLM interface using Ollama — supports concurrent tool execution.

The key pattern: when we detect a tool intent (e.g. "weather"), we fire the
API call in a background thread AND start the LLM streaming simultaneously.
The LLM generates a bridge phrase while the tool runs, then incorporates
the tool result into the rest of its response. Zero dead air.
"""

from __future__ import annotations

import concurrent.futures

import ollama

from vox.config import OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT
from vox.tools import TOOL_DEFINITIONS, DetectedIntent, detect_intent, execute_tool, validate_tool_call

# Conversation history
_history: list[dict] = []
MAX_HISTORY = 20


def _get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def chat(
    user_message: str,
    model_override: str | None = None,
    on_chunk: callable | None = None,
) -> str:
    """Send a message to the LLM. Uses concurrent tool execution when possible.

    Args:
        user_message: What the user said.
        model_override: Override the default Ollama model.
        on_chunk: Optional callback receiving text chunks as they stream in.
                  Signature: on_chunk(text: str) -> None
                  Used by TTS to start speaking before the full response is ready.

    Returns:
        Full response text.
    """
    model = model_override or OLLAMA_MODEL

    _history.append({"role": "user", "content": user_message})
    while len(_history) > MAX_HISTORY:
        _history.pop(0)

    # Fast intent detection (~1ms, regex-based)
    intent = detect_intent(user_message)

    if intent is not None:
        response = _chat_with_concurrent_tool(model, intent, on_chunk)
    else:
        response = _chat_standard(model, on_chunk)

    _history.append({"role": "assistant", "content": response})
    return response


def _chat_with_concurrent_tool(
    model: str,
    intent: DetectedIntent,
    on_chunk: callable | None,
) -> str:
    """Fire tool call in background, stream LLM bridge phrase, then merge results.

    Timeline:
      t=0ms   → Tool API call starts in background thread
      t=0ms   → LLM starts streaming bridge phrase to TTS
      t=200ms → User hears VOX start talking
      t=500ms → Tool result arrives (weather, time, etc.)
      t=600ms → LLM gets tool result, continues streaming with real data
    """
    client = _get_client()

    # 1. Fire tool in background thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        tool_future = executor.submit(execute_tool, intent.tool_name, intent.args)

        # 2. Stream the bridge phrase immediately via on_chunk
        if on_chunk:
            on_chunk(intent.bridge_phrase + " ")
        print(f"[LLM] Bridge: {intent.bridge_phrase}")
        print(f"[LLM] Tool '{intent.tool_name}' running concurrently...")

        # 3. Wait for tool result (usually <1s for API calls)
        tool_result = tool_future.result(timeout=10)
        print(f"[LLM] Tool result: {tool_result[:100]}...")

    # 4. Now ask the LLM to synthesize a natural response with the real data
    _history.append({"role": "tool", "content": tool_result})

    synthesis_prompt = (
        f"You already told the user: \"{intent.bridge_phrase}\"\n"
        f"Now continue your response naturally using this data:\n{tool_result}\n"
        f"Do NOT repeat the bridge phrase. Just continue the sentence with the answer. "
        f"Be concise — 1-3 sentences."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history,
        {"role": "system", "content": synthesis_prompt},
    ]

    # Stream the synthesis
    full_response = intent.bridge_phrase + " "
    stream = client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"].get("content", "")
        if token:
            full_response += token
            if on_chunk:
                on_chunk(token)

    print(f"[LLM] Full response: {full_response}")
    return full_response


def _chat_standard(model: str, on_chunk: callable | None) -> str:
    """Standard chat — no concurrent tools. Supports multi-step tool chains."""
    client = _get_client()
    current_msg = _history[-1]["content"] if _history else ""
    max_tool_rounds = 3  # prevent infinite tool loops

    for round_num in range(max_tool_rounds):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history]

        response = client.chat(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS if TOOL_DEFINITIONS else None,
        )

        msg = response["message"]

        if not msg.get("tool_calls"):
            break  # No tool calls — we have our final response

        # Execute validated tool calls
        any_executed = False
        for tool_call in msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = tool_call["function"]["arguments"]

            if not validate_tool_call(fn_name, current_msg):
                print(f"[LLM] BLOCKED spurious tool call: {fn_name}({fn_args}) — not relevant to user message")
                continue

            print(f"[LLM] Tool call: {fn_name}({fn_args})")
            result = execute_tool(fn_name, fn_args)
            _history.append({"role": "tool", "content": str(result)})
            any_executed = True

        if not any_executed:
            # All tool calls were blocked — retry without tools
            print("[LLM] All tool calls blocked, responding without tools...")
            response = client.chat(model=model, messages=messages, tools=None)
            msg = response["message"]
            break

        # Loop back — LLM might want to call another tool (e.g., search then email)

    # Stream the final response
    reply = msg.get("content", "")
    if reply:
        if on_chunk:
            on_chunk(reply)
        print(f"[LLM] Response: {reply}")
        return reply

    # If LLM returned empty content with tool calls, get a streamed final answer
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history]
    full_response = ""
    stream = client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"].get("content", "")
        if token:
            full_response += token
            if on_chunk:
                on_chunk(token)
    print(f"[LLM] Response: {full_response}")
    return full_response


def clear_history() -> None:
    """Clear conversation history."""
    _history.clear()

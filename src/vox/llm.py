"""LLM interface using Ollama — supports concurrent tool execution.

The key pattern: when we detect a tool intent (e.g. "weather"), we fire the
API call in a background thread AND start the LLM streaming simultaneously.
The LLM generates a bridge phrase while the tool runs, then incorporates
the tool result into the rest of its response. Zero dead air.
"""

from __future__ import annotations

import concurrent.futures
import logging

import ollama

log = logging.getLogger(__name__)

from vox.config import OLLAMA_HOST, OLLAMA_MODEL, SYSTEM_PROMPT
from vox.tools import (
    TOOL_DEFINITIONS,
    DetectedIntent,
    detect_all_intents,
    execute_tool,
    validate_tool_call,
)

# Conversation history
_history: list[dict] = []
MAX_HISTORY = 20

# Per-tool timeouts — GPU-heavy tools need much longer than API calls
_TOOL_TIMEOUTS: dict[str, int] = {
    "generate_image": 300,  # model download + load + inference
}
_DEFAULT_TIMEOUT = 15


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
    intents = detect_all_intents(user_message)

    if intents:
        log.info("Routing to concurrent tool path: %s", [i.tool_name for i in intents])
        response = _chat_with_concurrent_tool(model, intents, on_chunk)
    else:
        log.info("Routing to standard LLM chat (no intents detected)")
        response = _chat_standard(model, on_chunk)

    _history.append({"role": "assistant", "content": response})
    return response


def _chat_with_concurrent_tool(
    model: str,
    intents: list[DetectedIntent],
    on_chunk: callable | None,
) -> str:
    """Fire tool calls and stream LLM synthesis. Supports chaining multiple tools.

    Single tool timeline:
      t=0ms   → Tool API call starts in background thread
      t=0ms   → LLM starts streaming bridge phrase to TTS
      t=200ms → User hears VOX start talking
      t=500ms → Tool result arrives (weather, time, etc.)
      t=600ms → LLM gets tool result, continues streaming with real data

    Chained tools (e.g., search + email):
      First tool runs concurrently with bridge phrase, then subsequent tools
      run sequentially using the previous tool's result as context.
    """
    client = _get_client()
    primary = intents[0]
    chained = intents[1:]

    # 1. Fire primary tool in background thread
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        tool_future = executor.submit(execute_tool, primary.tool_name, primary.args)

        # 2. Stream the bridge phrase immediately via on_chunk
        if on_chunk:
            on_chunk(primary.bridge_phrase + " ")
        log.info("Bridge: %s", primary.bridge_phrase)
        log.info("Tool '%s' running concurrently with args=%s", primary.tool_name, primary.args)

        # 3. Wait for tool result (API calls ~1s, image gen ~5-60s)
        timeout = _TOOL_TIMEOUTS.get(primary.tool_name, _DEFAULT_TIMEOUT)
        tool_result = tool_future.result(timeout=timeout)
        log.info("Tool '%s' result (%d chars): %s", primary.tool_name, len(tool_result), tool_result[:200])

    # 4. Run chained tools sequentially (e.g., email the search results)
    chained_results = []
    for chained_intent in chained:
        # Enrich chained tool args with context from the primary result
        args = dict(chained_intent.args)
        if chained_intent.tool_name == "send_email":
            if not args.get("subject"):
                args["subject"] = f"VOX: {primary.args.get('query', primary.args.get('prompt', 'Results'))}"
            if not args.get("body"):
                args["body"] = tool_result
            # Attach generated image file if chaining from generate_image
            if primary.tool_name == "generate_image":
                import re as _re
                path_match = _re.search(r"saved to (.+)$", tool_result)
                if path_match:
                    args["attachments"] = [path_match.group(1)]
                    args["body"] = "Here's the image you requested."

        log.info("Chained tool: %s args=%s", chained_intent.tool_name, {k: v for k, v in args.items() if k != "body"})
        chained_result = execute_tool(chained_intent.tool_name, args)
        chained_results.append((chained_intent.tool_name, chained_result))
        log.info("Chained result (%d chars): %s", len(chained_result), chained_result[:200])

    # 5. Build history with all tool results
    _history.append({"role": "assistant", "content": primary.bridge_phrase})
    _history.append({"role": "tool", "content": tool_result})
    for tool_name, result in chained_results:
        _history.append({"role": "tool", "content": result})

    # 6. Ask the LLM to synthesize a natural response with all tool data
    all_results = f"Primary tool ({primary.tool_name}):\n{tool_result}"
    for tool_name, result in chained_results:
        all_results += f"\n\nChained tool ({tool_name}):\n{result}"

    synthesis_msg = (
        f"[SYSTEM INSTRUCTION — do not read this aloud]\n"
        f"The tools returned this data:\n{all_results}\n\n"
        f"Now respond to the user's original question using the data above. "
        f"Do NOT say \"{primary.bridge_phrase}\" again — you already said that. "
        f"Just give the answer naturally. Be concise — 1-3 sentences."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history,
        {"role": "user", "content": synthesis_msg},
    ]

    # Stream the synthesis
    full_response = primary.bridge_phrase + " "
    stream = client.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token = chunk["message"].get("content", "")
        if token:
            full_response += token
            if on_chunk:
                on_chunk(token)

    # If synthesis returned nothing, just format the tool result directly
    if full_response.strip() == primary.bridge_phrase.strip():
        log.warning("LLM synthesis empty — using tool result directly")
        full_response = primary.bridge_phrase + " " + tool_result.split("\n")[0]
        if on_chunk:
            on_chunk(tool_result.split("\n")[0])

    log.info("Full response (%d chars): %s", len(full_response), full_response[:200])
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
                log.warning("BLOCKED spurious tool call: %s(%s) — not relevant to: %s", fn_name, fn_args, current_msg[:80])
                continue

            log.info("LLM tool call: %s(%s)", fn_name, fn_args)
            result = execute_tool(fn_name, fn_args)
            _history.append({"role": "tool", "content": str(result)})
            any_executed = True

        if not any_executed:
            # All tool calls were blocked — retry without tools
            log.warning("All LLM tool calls blocked, responding without tools")
            response = client.chat(model=model, messages=messages, tools=None)
            msg = response["message"]
            break

        # Loop back — LLM might want to call another tool (e.g., search then email)

    # Stream the final response
    reply = msg.get("content", "")
    if reply:
        if on_chunk:
            on_chunk(reply)
        log.info("LLM response (%d chars): %s", len(reply), reply[:200])
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
    log.info("LLM response (%d chars): %s", len(full_response), full_response[:200])
    return full_response


def clear_history() -> None:
    """Clear conversation history."""
    _history.clear()

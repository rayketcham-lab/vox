"""MCP (Model Context Protocol) server for VOX.

Exposes VOX's capabilities as MCP tools so external agents (Claude, other
MCP clients) can interact with VOX. Also enables VOX to consume MCP servers
for calendar, smart home, Spotify, etc.

Server mode:  python -m vox.mcp_server
Client mode:  Use connect_server() to add external MCP tool servers.

Protocol: https://modelcontextprotocol.io/specification/2025-11-25
"""

from __future__ import annotations

import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

log = logging.getLogger(__name__)

server = Server("vox")


# ---------------------------------------------------------------------------
# Tools — expose VOX capabilities to MCP clients
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise VOX's tools to MCP clients."""
    return [
        Tool(
            name="chat",
            description=(
                "Send a message to VOX and get a response. "
                "VOX is a local AI assistant with persona, memory, and tool-calling capabilities."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "The message to send to VOX"},
                },
                "required": ["message"],
            },
        ),
        Tool(
            name="remember",
            description="Store a fact in VOX's persistent memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember"},
                    "category": {
                        "type": "string",
                        "description": "Category (general, preference, date, person)",
                        "default": "general",
                    },
                },
                "required": ["fact"],
            },
        ),
        Tool(
            name="recall",
            description="Search VOX's memory for facts matching a query. Uses both keyword and semantic search.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_memory",
            description="Semantic search over VOX's conversation history and stored facts using vector embeddings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "n_results": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="generate_image",
            description="Generate an image using Stable Diffusion XL. Returns the filename of the generated image.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Image generation prompt"},
                    "style": {"type": "string", "description": "Optional style tags"},
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="add_reminder",
            description="Set a timed reminder.",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder message"},
                    "minutes": {"type": "number", "description": "Minutes from now"},
                },
                "required": ["message", "minutes"],
            },
        ),
        Tool(
            name="add_todo",
            description="Add a task to the todo list.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task description"},
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="list_todos",
            description="List all active todo items.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="model_status",
            description="Check which AI models are currently loaded in GPU memory.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a VOX tool via MCP."""
    try:
        if name == "chat":
            from vox.llm import chat
            response = chat(arguments["message"])
            return [TextContent(type="text", text=response)]

        elif name == "remember":
            from vox.memory import remember
            result = remember(arguments["fact"], arguments.get("category", "general"))
            try:
                from vox.vector_memory import store_fact
                store_fact(arguments["fact"], arguments.get("category", "general"))
            except Exception:
                log.debug("Vector memory store failed in MCP remember")
            return [TextContent(type="text", text=result)]

        elif name == "recall":
            from vox.memory import recall
            result = recall(arguments.get("query", ""))
            return [TextContent(type="text", text=result)]

        elif name == "search_memory":
            from vox.vector_memory import search
            hits = search(arguments["query"], arguments.get("n_results", 5))
            if not hits:
                return [TextContent(type="text", text="No relevant memories found.")]
            lines = []
            for h in hits:
                meta = h["metadata"]
                dist = h["distance"]
                if meta.get("type") == "conversation":
                    user_msg = meta.get('user_message', '')[:100]
                    asst_msg = meta.get('assistant_response', '')[:100]
                    lines.append(f"[{dist:.2f}] User: {user_msg} → {asst_msg}")
                else:
                    lines.append(f"[{dist:.2f}] Fact: {h['document'][:200]}")
            return [TextContent(type="text", text="\n".join(lines))]

        elif name == "generate_image":
            from vox.tools import execute_tool
            result = execute_tool("generate_image", {
                "prompt": arguments["prompt"],
                "style": arguments.get("style", ""),
            })
            return [TextContent(type="text", text=result)]

        elif name == "add_reminder":
            from vox.reminders import add_reminder
            result = add_reminder(arguments["message"], arguments["minutes"])
            return [TextContent(type="text", text=result)]

        elif name == "add_todo":
            from vox.todos import add_todo
            result = add_todo(arguments["task"])
            return [TextContent(type="text", text=result)]

        elif name == "list_todos":
            from vox.todos import list_todos
            result = list_todos()
            return [TextContent(type="text", text=result)]

        elif name == "model_status":
            from vox import model_manager
            status = model_manager.status()
            vram = model_manager.vram_loaded()
            lines = [f"VRAM loaded: {vram}MB"]
            for name_, info in status.items():
                state = "LOADED" if info["loaded"] else "unloaded"
                lines.append(f"  {name_}: {state} ({info['vram_mb']}MB)")
            return [TextContent(type="text", text="\n".join(lines))]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        log.exception("MCP tool %s failed: %s", name, e)
        return [TextContent(type="text", text=f"Error: {e}")]


# ---------------------------------------------------------------------------
# Resources — expose VOX data to MCP clients
# ---------------------------------------------------------------------------

@server.list_resources()
async def list_resources() -> list[Resource]:
    """Advertise VOX data as MCP resources."""
    return [
        Resource(
            uri="vox://memory/facts",
            name="VOX Memory Facts",
            description="All stored memory facts (core memory)",
            mimeType="application/json",
        ),
        Resource(
            uri="vox://memory/stats",
            name="VOX Memory Stats",
            description="Memory system statistics",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a VOX resource."""
    import json

    if uri == "vox://memory/facts":
        from vox.memory import _load
        facts = _load()
        return json.dumps(facts, indent=2)

    elif uri == "vox://memory/stats":
        from vox.memory import _load
        from vox.vector_memory import count
        facts = _load()
        return json.dumps({
            "core_facts": len(facts),
            "vector_entries": count(),
        })

    return json.dumps({"error": f"Unknown resource: {uri}"})


# ---------------------------------------------------------------------------
# MCP Client — connect to external MCP tool servers
# ---------------------------------------------------------------------------

_mcp_clients: dict[str, object] = {}


async def connect_server(name: str, command: str, args: list[str] | None = None) -> list[str]:
    """Connect to an external MCP server and register its tools.

    Args:
        name: Friendly name (e.g., "spotify", "home-assistant")
        command: Server executable (e.g., "npx", "python")
        args: Command arguments (e.g., ["-y", "@modelcontextprotocol/server-spotify"])

    Returns:
        List of tool names available from the server.
    """
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=command, args=args or [])

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            _mcp_clients[name] = {"session": session, "tools": tool_names}
            log.info("Connected MCP server '%s': %d tools available", name, len(tool_names))
            return tool_names


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run():
    """Run the VOX MCP server (stdio transport)."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main():
    """CLI entry point for the MCP server."""
    import asyncio
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log.info("Starting VOX MCP server...")
    asyncio.run(run())


if __name__ == "__main__":
    main()

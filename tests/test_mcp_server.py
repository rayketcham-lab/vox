"""Tests for the MCP server — tool listing, dispatch, resources."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mcp", reason="mcp not installed")

# ---------------------------------------------------------------------------
# Stub out heavy VOX modules before importing mcp_server
# ---------------------------------------------------------------------------

_STUBBED_MODULES = [
    "vox.llm",
    "vox.memory",
    "vox.vector_memory",
    "vox.tools",
    "vox.reminders",
    "vox.todos",
    "vox.model_manager",
]

# Save original sys.modules state so we can restore after import
_saved_modules: dict[str, object] = {}
_stubs: dict[str, MagicMock] = {}
for _mod in _STUBBED_MODULES:
    if _mod in sys.modules:
        _saved_modules[_mod] = sys.modules[_mod]
    else:
        _stubs[_mod] = MagicMock()
        sys.modules[_mod] = _stubs[_mod]

from vox.mcp_server import call_tool, list_resources, list_tools, read_resource  # noqa: E402

# Restore original modules — prevents stub leaking into other test files
for _mod in _STUBBED_MODULES:
    if _mod in _saved_modules:
        sys.modules[_mod] = _saved_modules[_mod]
    elif _mod in _stubs:
        del sys.modules[_mod]


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

EXPECTED_TOOL_NAMES = [
    "chat",
    "remember",
    "recall",
    "search_memory",
    "generate_image",
    "add_reminder",
    "add_todo",
    "list_todos",
    "model_status",
]


@pytest.mark.asyncio
async def test_list_tools_returns_all_expected():
    tools = await list_tools()
    names = [t.name for t in tools]
    assert names == EXPECTED_TOOL_NAMES


@pytest.mark.asyncio
async def test_list_tools_count():
    tools = await list_tools()
    assert len(tools) == 9


@pytest.mark.asyncio
async def test_list_tools_have_descriptions():
    tools = await list_tools()
    for tool in tools:
        assert tool.description, f"Tool '{tool.name}' has no description"
        assert len(tool.description) > 10


@pytest.mark.asyncio
async def test_list_tools_have_input_schemas():
    tools = await list_tools()
    for tool in tools:
        assert tool.inputSchema is not None, f"Tool '{tool.name}' missing inputSchema"
        assert tool.inputSchema.get("type") == "object"


@pytest.mark.asyncio
async def test_chat_tool_requires_message():
    tools = await list_tools()
    chat_tool = next(t for t in tools if t.name == "chat")
    assert "message" in chat_tool.inputSchema["required"]


@pytest.mark.asyncio
async def test_remember_tool_requires_fact():
    tools = await list_tools()
    remember_tool = next(t for t in tools if t.name == "remember")
    assert "fact" in remember_tool.inputSchema["required"]
    # category is optional (has default)
    assert "category" not in remember_tool.inputSchema.get("required", [])


@pytest.mark.asyncio
async def test_add_reminder_requires_message_and_minutes():
    tools = await list_tools()
    reminder_tool = next(t for t in tools if t.name == "add_reminder")
    required = reminder_tool.inputSchema["required"]
    assert "message" in required
    assert "minutes" in required


# ---------------------------------------------------------------------------
# call_tool — dispatch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_call_tool_chat():
    with patch("vox.mcp_server.chat", create=True), \
         patch.dict(sys.modules, {"vox.llm": MagicMock()}):
        mock_chat = MagicMock(return_value="Hello there!")
        with patch.dict(sys.modules, {"vox.llm": MagicMock(chat=mock_chat)}):
            result = await call_tool("chat", {"message": "hi"})
    assert len(result) == 1
    assert result[0].type == "text"


@pytest.mark.asyncio
async def test_call_tool_remember():
    mock_remember = MagicMock(return_value="Remembered: sky is blue")
    mock_store = MagicMock()
    mock_memory = MagicMock(remember=mock_remember)
    mock_vmem = MagicMock(store_fact=mock_store)
    with patch.dict(sys.modules, {"vox.memory": mock_memory, "vox.vector_memory": mock_vmem}):
        result = await call_tool("remember", {"fact": "sky is blue"})
    assert len(result) == 1
    assert result[0].type == "text"
    assert "Remembered" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_remember_with_category():
    mock_remember = MagicMock(return_value="Remembered: birthday March 5")
    mock_memory = MagicMock(remember=mock_remember)
    mock_vmem = MagicMock(store_fact=MagicMock())
    with patch.dict(sys.modules, {"vox.memory": mock_memory, "vox.vector_memory": mock_vmem}):
        result = await call_tool("remember", {"fact": "birthday March 5", "category": "date"})
    assert len(result) == 1
    mock_remember.remember.assert_not_called()  # called via from-import


@pytest.mark.asyncio
async def test_call_tool_remember_default_category():
    mock_remember = MagicMock(return_value="ok")
    mock_memory = MagicMock(remember=mock_remember)
    mock_vmem = MagicMock(store_fact=MagicMock())
    with patch.dict(sys.modules, {"vox.memory": mock_memory, "vox.vector_memory": mock_vmem}):
        result = await call_tool("remember", {"fact": "test fact"})
    assert result[0].type == "text"


@pytest.mark.asyncio
async def test_call_tool_recall():
    mock_recall = MagicMock(return_value="Found: sky is blue")
    mock_memory = MagicMock(recall=mock_recall)
    with patch.dict(sys.modules, {"vox.memory": mock_memory}):
        result = await call_tool("recall", {"query": "sky"})
    assert len(result) == 1
    assert "Found" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_recall_empty_query():
    mock_recall = MagicMock(return_value="No results")
    mock_memory = MagicMock(recall=mock_recall)
    with patch.dict(sys.modules, {"vox.memory": mock_memory}):
        result = await call_tool("recall", {})
    assert len(result) == 1


@pytest.mark.asyncio
async def test_call_tool_search_memory_with_hits():
    hits = [
        {
            "metadata": {"type": "conversation", "user_message": "hello", "assistant_response": "hi there"},
            "distance": 0.12,
            "document": "hello",
        },
        {
            "metadata": {"type": "fact", "category": "general"},
            "distance": 0.45,
            "document": "The sky is blue",
        },
    ]
    mock_vmem = MagicMock(search=MagicMock(return_value=hits))
    with patch.dict(sys.modules, {"vox.vector_memory": mock_vmem}):
        result = await call_tool("search_memory", {"query": "sky"})
    assert len(result) == 1
    text = result[0].text
    assert "[0.12]" in text
    assert "User:" in text
    assert "[0.45]" in text
    assert "Fact:" in text


@pytest.mark.asyncio
async def test_call_tool_search_memory_no_hits():
    mock_vmem = MagicMock(search=MagicMock(return_value=[]))
    with patch.dict(sys.modules, {"vox.vector_memory": mock_vmem}):
        result = await call_tool("search_memory", {"query": "nonexistent"})
    assert result[0].text == "No relevant memories found."


@pytest.mark.asyncio
async def test_call_tool_search_memory_default_n_results():
    mock_search = MagicMock(return_value=[])
    mock_vmem = MagicMock(search=mock_search)
    with patch.dict(sys.modules, {"vox.vector_memory": mock_vmem}):
        await call_tool("search_memory", {"query": "test"})
    # The default n_results is 5
    mock_vmem.search.assert_called_once_with("test", 5)


@pytest.mark.asyncio
async def test_call_tool_generate_image():
    mock_execute = MagicMock(return_value="image_001.png")
    mock_tools = MagicMock(execute_tool=mock_execute)
    with patch.dict(sys.modules, {"vox.tools": mock_tools}):
        result = await call_tool("generate_image", {"prompt": "a sunset"})
    assert result[0].text == "image_001.png"


@pytest.mark.asyncio
async def test_call_tool_generate_image_with_style():
    mock_execute = MagicMock(return_value="styled.png")
    mock_tools = MagicMock(execute_tool=mock_execute)
    with patch.dict(sys.modules, {"vox.tools": mock_tools}):
        result = await call_tool("generate_image", {"prompt": "cat", "style": "anime"})
    assert result[0].text == "styled.png"


@pytest.mark.asyncio
async def test_call_tool_add_reminder():
    mock_add = MagicMock(return_value="Reminder set for 10 minutes")
    mock_reminders = MagicMock(add_reminder=mock_add)
    with patch.dict(sys.modules, {"vox.reminders": mock_reminders}):
        result = await call_tool("add_reminder", {"message": "take break", "minutes": 10})
    assert "Reminder" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_add_todo():
    mock_add = MagicMock(return_value="Added: buy groceries")
    mock_todos = MagicMock(add_todo=mock_add)
    with patch.dict(sys.modules, {"vox.todos": mock_todos}):
        result = await call_tool("add_todo", {"task": "buy groceries"})
    assert "Added" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_list_todos():
    mock_list = MagicMock(return_value="1. buy groceries\n2. walk dog")
    mock_todos = MagicMock(list_todos=mock_list)
    with patch.dict(sys.modules, {"vox.todos": mock_todos}):
        result = await call_tool("list_todos", {})
    assert "buy groceries" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_model_status():
    mock_status = MagicMock(return_value={
        "whisper": {"loaded": True, "vram_mb": 1500},
        "llm": {"loaded": True, "vram_mb": 10000},
        "tts": {"loaded": False, "vram_mb": 0},
    })
    mock_vram = MagicMock(return_value=11500)
    mock_mm = MagicMock(status=mock_status, vram_loaded=mock_vram)
    with patch.dict(sys.modules, {"vox.model_manager": mock_mm, "vox": MagicMock(model_manager=mock_mm)}):
        result = await call_tool("model_status", {})
    text = result[0].text
    assert "VRAM loaded: 11500MB" in text
    assert "whisper: LOADED" in text
    assert "tts: unloaded" in text


@pytest.mark.asyncio
async def test_call_tool_unknown_returns_error():
    result = await call_tool("totally_bogus_tool", {})
    assert len(result) == 1
    assert result[0].type == "text"
    assert "Unknown tool" in result[0].text
    assert "totally_bogus_tool" in result[0].text


@pytest.mark.asyncio
async def test_call_tool_exception_returns_error():
    mock_memory = MagicMock()
    mock_memory.recall = MagicMock(side_effect=RuntimeError("DB connection failed"))
    with patch.dict(sys.modules, {"vox.memory": mock_memory}):
        result = await call_tool("recall", {"query": "anything"})
    assert "Error:" in result[0].text
    assert "DB connection failed" in result[0].text


# ---------------------------------------------------------------------------
# list_resources
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_resources_returns_expected():
    resources = await list_resources()
    uris = [str(r.uri) for r in resources]
    assert "vox://memory/facts" in uris
    assert "vox://memory/stats" in uris


@pytest.mark.asyncio
async def test_list_resources_count():
    resources = await list_resources()
    assert len(resources) == 2


@pytest.mark.asyncio
async def test_list_resources_have_names():
    resources = await list_resources()
    for r in resources:
        assert r.name
        assert r.description


@pytest.mark.asyncio
async def test_list_resources_mime_types():
    resources = await list_resources()
    for r in resources:
        assert r.mimeType == "application/json"


# ---------------------------------------------------------------------------
# read_resource
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_read_resource_facts():
    fake_facts = [{"fact": "sky is blue", "category": "general"}]
    mock_memory = MagicMock(_load=MagicMock(return_value=fake_facts))
    with patch.dict(sys.modules, {"vox.memory": mock_memory}):
        raw = await read_resource("vox://memory/facts")
    data = json.loads(raw)
    assert data == fake_facts


@pytest.mark.asyncio
async def test_read_resource_stats():
    fake_facts = [{"fact": "a"}, {"fact": "b"}, {"fact": "c"}]
    mock_memory = MagicMock(_load=MagicMock(return_value=fake_facts))
    mock_vmem = MagicMock(count=MagicMock(return_value=42))
    with patch.dict(sys.modules, {"vox.memory": mock_memory, "vox.vector_memory": mock_vmem}):
        raw = await read_resource("vox://memory/stats")
    data = json.loads(raw)
    assert data["core_facts"] == 3
    assert data["vector_entries"] == 42


@pytest.mark.asyncio
async def test_read_resource_unknown_uri():
    raw = await read_resource("vox://does/not/exist")
    data = json.loads(raw)
    assert "error" in data
    assert "Unknown resource" in data["error"]


@pytest.mark.asyncio
async def test_read_resource_returns_valid_json():
    """All resource reads must return valid JSON."""
    mock_memory = MagicMock(_load=MagicMock(return_value=[]))
    mock_vmem = MagicMock(count=MagicMock(return_value=0))
    with patch.dict(sys.modules, {"vox.memory": mock_memory, "vox.vector_memory": mock_vmem}):
        for uri in ["vox://memory/facts", "vox://memory/stats", "vox://bogus"]:
            raw = await read_resource(uri)
            # Should not raise
            json.loads(raw)

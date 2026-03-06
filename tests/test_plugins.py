"""Tests for the plugin system — loading, registration, and listing."""

import re
import textwrap

from vox.plugins import (
    PluginTool,
    _discover_plugins,
    _load_plugin,
    _loaded_plugins,
    get_loaded,
    list_plugins,
    register_with_tools,
)

# ---------------------------------------------------------------------------
# PluginTool dataclass
# ---------------------------------------------------------------------------

def test_plugin_tool_basic():
    tool = PluginTool(
        name="test_tool",
        description="A test tool",
        handler=lambda **kw: "ok",
    )
    assert tool.name == "test_tool"
    assert tool.description == "A test tool"
    assert tool._compiled_pattern is None
    assert tool._compiled_validator is None


def test_plugin_tool_with_pattern():
    tool = PluginTool(
        name="test_tool",
        description="A test tool",
        handler=lambda **kw: "ok",
        pattern=r"\b(hello)\b",
    )
    assert tool._compiled_pattern is not None
    assert tool._compiled_pattern.search("hello world")
    assert tool._compiled_pattern.flags & re.IGNORECASE


def test_plugin_tool_with_validator():
    tool = PluginTool(
        name="test_tool",
        description="A test tool",
        handler=lambda **kw: "ok",
        validator_pattern=r"\b(validate)\b",
    )
    assert tool._compiled_validator is not None
    assert tool._compiled_validator.search("validate this")


def test_plugin_tool_default_parameters():
    tool = PluginTool(
        name="t",
        description="d",
        handler=lambda **kw: "",
    )
    assert tool.parameters == {"type": "object", "properties": {}, "required": []}


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------

def test_discover_plugins_nonexistent_dirs():
    """If plugin dirs don't exist, discover returns empty list."""
    # Default dirs may or may not exist; just verify it returns a list
    result = _discover_plugins()
    assert isinstance(result, list)


def test_discover_plugins_skips_underscored(tmp_path):
    """Files starting with _ should be skipped."""
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "_hidden.py").write_text("# hidden")
    (plugin_dir / "visible.py").write_text("# visible")

    from vox import plugins
    original = plugins._PLUGIN_DIRS[:]
    try:
        plugins._PLUGIN_DIRS[:] = [plugin_dir]
        found = _discover_plugins()
        names = [f.name for f in found]
        assert "visible.py" in names
        assert "_hidden.py" not in names
    finally:
        plugins._PLUGIN_DIRS[:] = original


# ---------------------------------------------------------------------------
# Plugin loading
# ---------------------------------------------------------------------------

def test_load_plugin_valid(tmp_path):
    """A plugin with register() returning PluginTool should load."""
    plugin_file = tmp_path / "good_plugin.py"
    plugin_file.write_text(textwrap.dedent("""\
        from vox.plugins import PluginTool

        def register():
            return [
                PluginTool(
                    name="good_tool",
                    description="A good tool",
                    handler=lambda **kw: "good",
                )
            ]
    """))
    tools = _load_plugin(plugin_file)
    assert len(tools) == 1
    assert tools[0].name == "good_tool"
    assert tools[0].handler() == "good"


def test_load_plugin_no_register(tmp_path):
    """A plugin without register() should return empty."""
    plugin_file = tmp_path / "bad_plugin.py"
    plugin_file.write_text("x = 1\n")
    tools = _load_plugin(plugin_file)
    assert tools == []


def test_load_plugin_register_returns_single(tmp_path):
    """register() returning a single tool (not list) should still work."""
    plugin_file = tmp_path / "single_plugin.py"
    plugin_file.write_text(textwrap.dedent("""\
        from vox.plugins import PluginTool

        def register():
            return PluginTool(
                name="single",
                description="Single tool",
                handler=lambda **kw: "one",
            )
    """))
    tools = _load_plugin(plugin_file)
    assert len(tools) == 1
    assert tools[0].name == "single"


def test_load_plugin_syntax_error(tmp_path):
    """A plugin with a syntax error should return empty, not crash."""
    plugin_file = tmp_path / "broken.py"
    plugin_file.write_text("def register(\n")  # syntax error
    tools = _load_plugin(plugin_file)
    assert tools == []


# ---------------------------------------------------------------------------
# Registration with tools module
# ---------------------------------------------------------------------------

def test_register_with_tools_mock():
    """register_with_tools should populate registry, patterns, validators, and definitions."""

    class MockToolsModule:
        _TOOL_REGISTRY = {}
        _INTENT_PATTERNS = []
        _TOOL_VALIDATORS = {}
        TOOL_DEFINITIONS = []

    # Clear loaded plugins so load_all picks up fresh
    _loaded_plugins.clear()

    # Create a fake plugin tool
    tool = PluginTool(
        name="mock_tool",
        description="Mock tool",
        handler=lambda **kw: "mocked",
        pattern=r"\b(mock)\b",
        validator_pattern=r"\b(mock)\b",
    )

    # Monkey-patch load_all to return our tool
    import vox.plugins as pm
    original_load = pm.load_all

    def fake_load():
        _loaded_plugins["mock"] = {"path": "fake", "tools": ["mock_tool"]}
        return [tool]

    pm.load_all = fake_load
    try:
        count = register_with_tools(MockToolsModule)
        assert count == 1
        assert "mock_tool" in MockToolsModule._TOOL_REGISTRY
        assert len(MockToolsModule._INTENT_PATTERNS) == 1
        assert "mock_tool" in MockToolsModule._TOOL_VALIDATORS
        assert len(MockToolsModule.TOOL_DEFINITIONS) == 1
        assert MockToolsModule.TOOL_DEFINITIONS[0]["function"]["name"] == "mock_tool"
    finally:
        pm.load_all = original_load
        _loaded_plugins.clear()


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_list_plugins_empty():
    _loaded_plugins.clear()
    result = list_plugins()
    assert "No plugins loaded" in result


def test_list_plugins_with_data():
    _loaded_plugins.clear()
    _loaded_plugins["test_plugin"] = {"path": "/fake/test_plugin.py", "tools": ["tool_a", "tool_b"]}
    result = list_plugins()
    assert "1 plugin(s) loaded" in result
    assert "test_plugin" in result
    assert "tool_a, tool_b" in result
    _loaded_plugins.clear()


def test_get_loaded():
    _loaded_plugins.clear()
    _loaded_plugins["x"] = {"path": "y", "tools": ["z"]}
    loaded = get_loaded()
    assert loaded == {"x": {"path": "y", "tools": ["z"]}}
    # Verify top-level is a copy (adding keys doesn't affect original)
    loaded["new"] = {}
    assert "new" not in _loaded_plugins
    _loaded_plugins.clear()

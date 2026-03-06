"""Plugin system — dynamically load tools from plugin directories.

Plugins are Python files dropped into:
  1. ~/.vox/plugins/     (user plugins)
  2. <project>/plugins/  (project plugins)

Each plugin is a Python module that defines a `register()` function
returning a list of tool definitions. Example plugin:

    # ~/.vox/plugins/my_tool.py
    from vox.plugins import PluginTool

    def register():
        return [
            PluginTool(
                name="my_tool",
                description="Does something useful",
                pattern=r"\\b(do something)\\b",
                handler=lambda **kwargs: "Done!",
                parameters={"type": "object", "properties": {}, "required": []},
            )
        ]
"""

from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Plugin directories
_PLUGIN_DIRS: list[Path] = [
    Path.home() / ".vox" / "plugins",
    Path(__file__).parent.parent.parent / "plugins",
]

# Loaded plugins
_loaded_plugins: dict[str, dict] = {}


@dataclass
class PluginTool:
    """A tool definition from a plugin."""
    name: str
    description: str
    handler: callable
    pattern: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}, "required": []})
    validator_pattern: str = ""
    _compiled_pattern: re.Pattern | None = field(default=None, init=False, repr=False)
    _compiled_validator: re.Pattern | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern, re.IGNORECASE)
        if self.validator_pattern:
            self._compiled_validator = re.compile(self.validator_pattern, re.IGNORECASE)


def _discover_plugins() -> list[Path]:
    """Find all plugin files in plugin directories."""
    plugins = []
    for plugin_dir in _PLUGIN_DIRS:
        if plugin_dir.exists() and plugin_dir.is_dir():
            for f in sorted(plugin_dir.glob("*.py")):
                if f.name.startswith("_"):
                    continue
                plugins.append(f)
    return plugins


def _load_plugin(path: Path) -> list[PluginTool]:
    """Load a single plugin file and return its tool definitions."""
    try:
        spec = importlib.util.spec_from_file_location(f"vox_plugin_{path.stem}", str(path))
        if spec is None or spec.loader is None:
            log.warning("Failed to create module spec for %s", path)
            return []

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "register"):
            log.warning("Plugin %s has no register() function — skipping", path.name)
            return []

        tools = module.register()
        if not isinstance(tools, list):
            tools = [tools]

        valid_tools = [t for t in tools if isinstance(t, PluginTool)]
        if valid_tools:
            log.info("Loaded plugin %s: %d tool(s)", path.name, len(valid_tools))
        return valid_tools

    except Exception as e:
        log.error("Failed to load plugin %s: %s", path.name, e)
        return []


def load_all() -> list[PluginTool]:
    """Discover and load all plugins. Returns list of all plugin tools."""
    all_tools: list[PluginTool] = []
    plugin_files = _discover_plugins()

    for path in plugin_files:
        if path.stem in _loaded_plugins:
            continue
        tools = _load_plugin(path)
        if tools:
            _loaded_plugins[path.stem] = {
                "path": str(path),
                "tools": [t.name for t in tools],
            }
            all_tools.extend(tools)

    return all_tools


def get_loaded() -> dict[str, dict]:
    """Return info about loaded plugins."""
    return dict(_loaded_plugins)


def register_with_tools(tools_module) -> int:
    """Register all plugin tools with the main tools module.

    This connects plugin intent patterns, validators, and handlers
    to the core tool system.

    Args:
        tools_module: The vox.tools module

    Returns:
        Number of tools registered.
    """
    plugin_tools = load_all()
    count = 0

    for pt in plugin_tools:
        # Register handler
        if hasattr(tools_module, '_TOOL_REGISTRY'):
            tools_module._TOOL_REGISTRY[pt.name] = pt.handler

        # Register intent pattern
        if pt._compiled_pattern and hasattr(tools_module, '_INTENT_PATTERNS'):
            tools_module._INTENT_PATTERNS.append({
                "pattern": pt._compiled_pattern,
                "tool_name": pt.name,
                "extractor": lambda m, t: {},
                "bridge_phrase": f"Running {pt.name}...",
            })

        # Register validator
        if pt._compiled_validator and hasattr(tools_module, '_TOOL_VALIDATORS'):
            tools_module._TOOL_VALIDATORS[pt.name] = pt._compiled_validator

        # Register tool definition
        if hasattr(tools_module, 'TOOL_DEFINITIONS'):
            tools_module.TOOL_DEFINITIONS.append({
                "type": "function",
                "function": {
                    "name": pt.name,
                    "description": pt.description,
                    "parameters": pt.parameters,
                },
            })

        count += 1
        log.info("Registered plugin tool: %s", pt.name)

    return count


def list_plugins() -> str:
    """Human-readable list of loaded plugins and their tools."""
    if not _loaded_plugins:
        return "No plugins loaded."

    lines = []
    for name, info in _loaded_plugins.items():
        tools_str = ", ".join(info["tools"])
        lines.append(f"  {name}: {tools_str} ({info['path']})")

    return f"{len(_loaded_plugins)} plugin(s) loaded:\n" + "\n".join(lines)

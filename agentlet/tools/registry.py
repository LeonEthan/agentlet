"""Tool registry and built-in tool-name definitions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from types import MappingProxyType

from agentlet.tools.base import ApprovalCategory, Tool, ToolDefinition

BUILT_IN_TOOL_CATEGORIES = MappingProxyType(
    {
        "Read": "read_only",
        "Write": "mutating",
        "Edit": "mutating",
        "Bash": "exec",
        "Glob": "read_only",
        "Grep": "read_only",
        "WebSearch": "external_or_interrupt",
        "WebFetch": "external_or_interrupt",
        "AskUserQuestion": "external_or_interrupt",
    }
)
BUILT_IN_TOOL_NAMES = tuple(BUILT_IN_TOOL_CATEGORIES)


class DuplicateToolError(ValueError):
    """Raised when a tool name is registered more than once."""


class UnknownToolError(LookupError):
    """Raised when a requested tool name is not present in the registry."""


class ToolRegistry:
    """Register and resolve tools by their stable exposed name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register one tool definition, rejecting duplicates and mismatches."""

        definition = tool.definition
        existing = self._tools.get(definition.name)
        if existing is not None:
            raise DuplicateToolError(f"tool already registered: {definition.name}")

        expected_category = builtin_tool_category(definition.name)
        if (
            expected_category is not None
            and definition.approval_category != expected_category
        ):
            raise ValueError(
                "built-in tool approval category mismatch for "
                f"{definition.name}: expected {expected_category}, got "
                f"{definition.approval_category}"
            )

        self._tools[definition.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a tool by name or ``None`` when it is not registered."""

        return self._tools.get(name)

    def resolve(self, name: str) -> Tool:
        """Return a tool by name or raise when it is unknown."""

        tool = self.get(name)
        if tool is None:
            raise UnknownToolError(f"unknown tool: {name}")
        return tool

    def definition(self, name: str) -> ToolDefinition:
        """Return the registered definition for one tool."""

        return self.resolve(name).definition

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return registered tool definitions in registration order."""

        return tuple(tool.definition for tool in self._tools.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._tools

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)


def is_builtin_tool_name(name: str) -> bool:
    """Return whether ``name`` is one of the fixed built-in tool names."""

    return name in BUILT_IN_TOOL_CATEGORIES


def builtin_tool_category(name: str) -> ApprovalCategory | None:
    """Return the canonical approval category for a built-in tool name."""

    category = BUILT_IN_TOOL_CATEGORIES.get(name)
    if category is None:
        return None
    return category

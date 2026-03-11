from __future__ import annotations

"""Tool contracts and a minimal in-memory execution registry."""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agentlet.agent.context import ToolCall, ToolResult


@dataclass(frozen=True)
class ToolSpec:
    """Provider-facing description of a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_provider_dict(self) -> dict[str, Any]:
        """Convert the tool schema into the OpenAI-compatible function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool(Protocol):
    """Small tool interface used by the agent loop."""

    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, arguments: dict[str, Any]) -> str: ...


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot be found or its inputs are invalid."""

    pass


class ToolRegistry:
    """Register tools and execute them behind one narrow boundary."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register or replace a tool by its declared name."""
        self._tools[tool.spec.name] = tool

    def get_tool_schemas(self) -> list[ToolSpec]:
        """Expose provider-visible schemas for all registered tools."""
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        """Decode arguments, execute the tool, and normalize the returned result."""
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: {call.name}")

        try:
            # The provider boundary stores arguments as JSON text, so the registry
            # is the place where we validate that shape before invoking the tool.
            arguments = json.loads(call.arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(
                f"Invalid JSON for tool {call.name}: {exc.msg}"
            ) from exc

        if not isinstance(arguments, dict):
            raise ToolExecutionError(
                f"Tool {call.name} arguments must decode to an object."
            )

        result = await tool.execute(arguments)
        # Tool output is normalized back into a message payload so the agent loop
        # can append it to context without knowing tool-specific details.
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=result,
        )

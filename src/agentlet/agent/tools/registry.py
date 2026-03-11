from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from agentlet.agent.context import ToolCall, ToolResult


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_provider_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    async def execute(self, arguments: dict[str, Any]) -> str: ...


class ToolExecutionError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def get_tool_schemas(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: {call.name}")

        try:
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
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=result,
        )

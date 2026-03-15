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


@dataclass(frozen=True)
class ToolApprovalRequest:
    """Normalized approval request emitted before unsafe tool execution."""

    tool_name: str
    scope: str
    arguments: dict[str, Any]
    summary: str


class ToolApprovalHandler(Protocol):
    """Small approval interface used by the registry before unsafe tool calls."""

    async def approve(self, request: ToolApprovalRequest) -> bool: ...


def build_tool_result_content(payload: dict[str, Any]) -> str:
    """Build a consistent JSON text envelope for tool results.

    Built-in tools return JSON text, not free-form prose, so the model sees
    predictable keys and the CLI can parse or summarize payloads without
    inventing a second result model.
    """
    # Use compact separators to reduce token usage in model context
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ToolRegistry:
    """Register tools and execute them behind one narrow boundary."""

    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        approval_handler: ToolApprovalHandler | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._approval_handler = approval_handler
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register or replace a tool by its declared name."""
        self._tools[tool.spec.name] = tool

    def get_tool_schemas(self) -> list[ToolSpec]:
        """Expose provider-visible schemas for all registered tools."""
        return [tool.spec for tool in self._tools.values()]

    def get_tool_names(self) -> list[str]:
        """Return enabled tool names in registration order."""
        return list(self._tools.keys())

    def set_approval_handler(self, approval_handler: ToolApprovalHandler | None) -> None:
        """Install or clear the approval handler used for unsafe tool calls."""
        self._approval_handler = approval_handler

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

        approval_request = _build_approval_request(call.name, arguments)
        if approval_request is not None:
            if self._approval_handler is None:
                raise ToolExecutionError(
                    f"Tool {call.name} requires approval, but no approval handler is configured."
                )
            approved = await self._approval_handler.approve(approval_request)
            if not approved:
                raise ToolExecutionError(f"Tool {call.name} execution was not approved.")

        result = await tool.execute(arguments)
        # Tool output is normalized back into a message payload so the agent loop
        # can append it to context without knowing tool-specific details.
        return ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content=result,
        )


_APPROVAL_SCOPE_BY_TOOL = {
    "write": "write",
    "edit": "write",
    "bash": "bash",
    "web_search": "network",
    "web_fetch": "network",
}


def _build_approval_request(
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolApprovalRequest | None:
    """Build an approval request for tools that need explicit user consent."""
    scope = _APPROVAL_SCOPE_BY_TOOL.get(tool_name)
    if scope is None:
        return None

    if tool_name in {"write", "edit"}:
        target = str(arguments.get("path", ""))
    elif tool_name == "bash":
        target = str(arguments.get("command", ""))
    elif tool_name == "web_search":
        target = str(arguments.get("query", ""))
    else:
        target = str(arguments.get("url", ""))

    summary = f"{tool_name} {target}".strip()
    return ToolApprovalRequest(
        tool_name=tool_name,
        scope=scope,
        arguments=arguments,
        summary=summary,
    )

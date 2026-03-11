from __future__ import annotations

"""Pure in-memory message state for the agent runtime.

This module intentionally stays free of provider SDK imports and tool execution
logic. Its job is limited to holding normalized messages and building the next
provider request from that state.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    """Normalized representation of a provider-requested tool invocation."""

    id: str
    name: str
    arguments_json: str

    def to_provider_dict(self) -> dict[str, Any]:
        """Convert the internal tool call into the provider-facing payload shape."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass(frozen=True)
class ToolResult:
    """Tool execution output that can be appended as a follow-up message."""

    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True)
class Message:
    """Single normalized chat message used across the runtime."""

    role: Role
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    def to_provider_dict(self) -> dict[str, Any]:
        """Serialize the message into the provider's OpenAI-like message schema."""
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            payload["tool_calls"] = [call.to_provider_dict() for call in self.tool_calls]
        return payload


class Context:
    """Mutable conversation state used to assemble the next model request.

    The context owns history mutation only. It does not decide when to call the
    model, when to stop the loop, or how tools are executed.
    """

    def __init__(self, system_prompt: str, history: list[Message] | None = None) -> None:
        self.system_prompt = system_prompt
        self.history = list(history or [])

    def build_messages(self, user_input: str | None = None) -> list[Message]:
        """Return the full message list for the next provider call.

        When a new user input is provided we append it to history first, so the
        in-memory history remains the source of truth for subsequent turns.
        """
        if user_input:
            self.history.append(Message(role="user", content=user_input))
        return [Message(role="system", content=self.system_prompt), *self.history]

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        """Record an assistant step, including requested tool calls if present."""
        self.history.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=tuple(tool_calls or ()),
            )
        )

    def add_tool_result(self, result: ToolResult) -> None:
        """Append the tool output in the standard tool-message format."""
        self.history.append(
            Message(
                role="tool",
                content=result.content,
                name=result.name,
                tool_call_id=result.tool_call_id,
            )
        )

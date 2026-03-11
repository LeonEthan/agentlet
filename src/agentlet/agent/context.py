from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str

    def to_provider_dict(self) -> dict[str, Any]:
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
    tool_call_id: str
    name: str
    content: str


@dataclass(frozen=True)
class Message:
    role: Role
    content: str | None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    def to_provider_dict(self) -> dict[str, Any]:
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
    def __init__(self, system_prompt: str, history: list[Message] | None = None) -> None:
        self.system_prompt = system_prompt
        self.history = list(history or [])

    def build_messages(self, user_input: str | None = None) -> list[Message]:
        if user_input:
            self.history.append(Message(role="user", content=user_input))
        return [Message(role="system", content=self.system_prompt), *self.history]

    def add_assistant_message(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.history.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=tuple(tool_calls or ()),
            )
        )

    def add_tool_result(self, result: ToolResult) -> None:
        self.history.append(
            Message(
                role="tool",
                content=result.content,
                name=result.name,
                tool_call_id=result.tool_call_id,
            )
        )

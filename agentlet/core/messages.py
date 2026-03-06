"""Provider-agnostic message contracts for the core loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentlet.core.types import JSONObject

MessageRole = Literal["system", "user", "assistant", "tool"]
VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A normalized model-emitted request to invoke one tool."""

    id: str
    name: str
    arguments: JSONObject = field(default_factory=dict)
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")
        object.__setattr__(self, "arguments", dict(self.arguments))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ToolCall":
        return cls(
            id=str(payload["id"]),
            name=str(payload["name"]),
            arguments=dict(payload.get("arguments", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class Message:
    """A provider-neutral message exchanged between the loop and model."""

    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_MESSAGE_ROLES:
            raise ValueError(f"unsupported message role: {self.role}")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages may include tool calls")
        if self.tool_call_id is not None and self.role != "tool":
            raise ValueError("tool_call_id is only valid on tool messages")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("tool messages must include tool_call_id")

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "Message":
        tool_calls = tuple(
            ToolCall.from_dict(tool_call_payload)
            for tool_call_payload in payload.get("tool_calls", [])
        )
        return cls(
            role=payload["role"],
            content=str(payload.get("content", "")),
            name=payload.get("name"),
            tool_calls=tool_calls,
            tool_call_id=payload.get("tool_call_id"),
            metadata=dict(payload.get("metadata", {})),
        )

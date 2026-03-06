"""Provider-agnostic request and response schemas for model clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from agentlet.core.messages import Message
from agentlet.core.types import JSONObject, TokenUsage

if TYPE_CHECKING:
    from agentlet.tools.base import ToolDefinition

ToolChoiceMode = Literal["auto", "none", "required", "tool"]


@dataclass(frozen=True, slots=True)
class ModelToolDefinition:
    """Model-facing tool schema without runtime approval policy."""

    name: str
    description: str
    input_schema: JSONObject
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError("tool description must not be empty")
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ModelToolDefinition":
        return cls(
            name=str(payload["name"]),
            description=str(payload["description"]),
            input_schema=dict(payload.get("input_schema", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_tool_definition(
        cls,
        definition: "ToolDefinition",
    ) -> "ModelToolDefinition":
        return cls(
            name=definition.name,
            description=definition.description,
            input_schema=definition.input_schema,
            metadata=definition.metadata,
        )


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """How the caller wants the model to use available tools."""

    mode: ToolChoiceMode = "auto"
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "tool" and not self.tool_name:
            raise ValueError("tool_name is required when mode='tool'")
        if self.mode != "tool" and self.tool_name is not None:
            raise ValueError("tool_name is only valid when mode='tool'")

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ToolChoice":
        return cls(
            mode=payload.get("mode", "auto"),
            tool_name=payload.get("tool_name"),
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Normalized input passed from the core loop to a model client."""

    messages: tuple[Message, ...]
    tools: tuple[ModelToolDefinition, ...] = field(default_factory=tuple)
    tool_choice: ToolChoice | None = None
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.messages:
            raise ValueError("messages must not be empty")
        if self.tool_choice is None:
            return
        available_tool_names = {tool.name for tool in self.tools}
        if self.tool_choice.mode in {"required", "tool"} and not self.tools:
            raise ValueError("tool_choice requires at least one tool definition")
        if (
            self.tool_choice.mode == "tool"
            and self.tool_choice.tool_name not in available_tool_names
        ):
            raise ValueError("tool_choice.tool_name must match an available tool")

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ModelRequest":
        return cls(
            messages=tuple(
                Message.from_dict(message_payload)
                for message_payload in payload.get("messages", [])
            ),
            tools=tuple(
                ModelToolDefinition.from_dict(tool_payload)
                for tool_payload in payload.get("tools", [])
            ),
            tool_choice=(
                ToolChoice.from_dict(payload["tool_choice"])
                if payload.get("tool_choice") is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized model output returned to the core loop."""

    message: Message
    finish_reason: str
    usage: TokenUsage | None = None
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.message.role != "assistant":
            raise ValueError("model responses must contain an assistant message")
        if not self.finish_reason:
            raise ValueError("finish_reason must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ModelResponse":
        usage_payload = payload.get("usage")
        return cls(
            message=Message.from_dict(payload["message"]),
            finish_reason=str(payload["finish_reason"]),
            usage=(
                TokenUsage.from_dict(usage_payload)
                if usage_payload is not None
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )

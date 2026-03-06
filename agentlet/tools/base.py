"""Shared tool contracts used by the agent loop and tool registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from agentlet.core.types import InterruptMetadata, JSONObject

ApprovalCategory = Literal[
    "read_only",
    "mutating",
    "exec",
    "external_or_interrupt",
]
VALID_APPROVAL_CATEGORIES = {
    "read_only",
    "mutating",
    "exec",
    "external_or_interrupt",
}


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Static description exposed to the model and registry."""

    name: str
    description: str
    input_schema: JSONObject
    approval_category: ApprovalCategory
    metadata: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if not self.description:
            raise ValueError("tool description must not be empty")
        if self.approval_category not in VALID_APPROVAL_CATEGORIES:
            raise ValueError(
                f"unsupported approval category: {self.approval_category}"
            )
        object.__setattr__(self, "input_schema", dict(self.input_schema))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Single normalized result type for every built-in tool."""

    output: str
    metadata: JSONObject | None = None
    is_error: bool = False
    interrupt: bool = False

    def __post_init__(self) -> None:
        if self.metadata is not None:
            object.__setattr__(self, "metadata", dict(self.metadata))
        interrupt_payload = None if self.metadata is None else self.metadata.get("interrupt")
        if self.interrupt and interrupt_payload is None:
            raise ValueError("interrupt results must include metadata['interrupt']")
        if not self.interrupt and interrupt_payload is not None:
            raise ValueError("metadata['interrupt'] requires interrupt=True")
        if interrupt_payload is not None:
            if not isinstance(interrupt_payload, dict):
                raise ValueError("metadata['interrupt'] must be a mapping")
            InterruptMetadata.from_dict(interrupt_payload)

    @classmethod
    def error(
        cls,
        output: str,
        metadata: JSONObject | None = None,
    ) -> "ToolResult":
        return cls(output=output, metadata=metadata, is_error=True)

    @classmethod
    def interrupt_result(
        cls,
        output: str,
        interrupt: InterruptMetadata,
        metadata: JSONObject | None = None,
    ) -> "ToolResult":
        combined_metadata = dict(metadata or {})
        combined_metadata["interrupt"] = interrupt.as_dict()
        return cls(output=output, metadata=combined_metadata, interrupt=True)


@runtime_checkable
class Tool(Protocol):
    """Runtime-checkable protocol for built-in tools."""

    @property
    def definition(self) -> ToolDefinition:
        """Return the static tool definition exposed to the model."""

    def execute(self, arguments: JSONObject) -> ToolResult:
        """Execute the tool with validated JSON-like arguments."""

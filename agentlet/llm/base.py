"""Provider interface contracts."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentlet.llm.schemas import ModelRequest, ModelResponse


@runtime_checkable
class ModelClient(Protocol):
    """Provider-agnostic model interface used by the core loop."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Produce the next assistant turn for a normalized request."""

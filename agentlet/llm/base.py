"""Provider interface contracts."""

from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from agentlet.llm.schemas import ModelRequest, ModelResponse

StreamHandler = Callable[[str], None]


@runtime_checkable
class ModelClient(Protocol):
    """Provider-agnostic model interface used by the core loop."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Produce the next assistant turn for a normalized request."""

    def complete_stream(
        self,
        request: ModelRequest,
        handler: StreamHandler,
    ) -> ModelResponse:
        """Produce the next assistant turn with streaming output.

        The handler is called with each text chunk as it arrives from the model.
        Returns the complete response after streaming finishes."""

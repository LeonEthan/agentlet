"""User interaction boundary for approvals and interrupt/resume flow."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentlet.runtime.events import (
    ApprovalRequest,
    ApprovalResponse,
    RuntimeEvent,
    UserQuestionRequest,
)


@runtime_checkable
class UserIO(Protocol):
    """App-layer boundary for human interaction without transport details."""

    def emit_event(self, event: RuntimeEvent) -> None:
        """Observe a runtime event for rendering, logging, or persistence."""

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Synchronously collect a user decision for an approval gate."""

    def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
        """Surface a structured question and return control to the app for resume."""

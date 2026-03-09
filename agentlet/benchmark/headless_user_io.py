"""Headless UserIO for benchmark execution without TTY."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentlet.core.interrupts import (
    VALID_APPROVAL_DECISIONS,
    ApprovalDecisionValue,
    ApprovalRequest,
    ApprovalResponse,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.runtime.events import RuntimeEvent


@dataclass
class HeadlessUserIO:
    """Non-interactive UserIO for containerized benchmark execution.

    - Auto-approves all operations based on configurable policy
    - Auto-selects first option for question interrupts
    - Logs all events to JSONL for post-run analysis
    """

    # Configuration
    auto_approve_all: bool = True  # If False, only auto-approves read_only
    default_question_response: str | None = None  # Override default option selection

    # State tracking
    events: list[RuntimeEvent] = field(default_factory=list)
    event_log_path: Path | None = None
    _log_file: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.event_log_path:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.event_log_path, "a")

    def close(self) -> None:
        """Close the log file if open."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def emit_event(self, event: RuntimeEvent) -> None:
        """Log event to memory and optionally to file."""
        self.events.append(event)
        if self._log_file is not None:
            self._log_file.write(json.dumps({
                "kind": event.kind,
                "payload": event.payload.as_dict(),
            }) + "\n")

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Auto-approve based on policy."""
        decision: ApprovalDecisionValue = (
            "approved" if self.auto_approve_all else "rejected"
        )
        if not self.auto_approve_all and request.approval_category == "read_only":
            decision = "approved"

        return ApprovalResponse(
            request_id=request.request_id,
            decision=decision,
        )

    def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
        """Record question interrupt."""

    def resolve_question_interrupt(
        self,
        request: UserQuestionRequest,
    ) -> UserQuestionResponse:
        """Auto-select first option or use configured default."""
        selected = self.default_question_response
        if selected is None and request.options:
            selected = request.options[0].value

        return UserQuestionResponse(
            request_id=request.request_id,
            selected_option=selected,
        )

    def get_trajectory_summary(self) -> dict[str, Any]:
        """Generate summary for Harbor AgentContext."""
        approval_count = sum(
            1 for e in self.events if e.kind == "approval_requested"
        )
        question_count = sum(
            1 for e in self.events if e.kind == "question_interrupted"
        )
        return {
            "total_events": len(self.events),
            "approval_requests": approval_count,
            "question_requests": question_count,
            "event_kinds": [e.kind for e in self.events],
        }



"""Runtime-facing event envelopes for approvals, interrupts, and resumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agentlet.core.interrupts import (
    ApprovalRequest,
    ApprovalResponse,
    ResumeRequest,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.core.types import JSONObject, deep_copy_json_object

RuntimeEventKind = Literal["approval_requested", "question_interrupted", "resumed"]
VALID_RUNTIME_EVENT_KINDS = {
    "approval_requested",
    "question_interrupted",
    "resumed",
}


EventPayload = ApprovalRequest | UserQuestionRequest | ResumeRequest


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Stable runtime event envelope used across transports and apps."""

    kind: RuntimeEventKind
    payload: EventPayload

    def __post_init__(self) -> None:
        if self.kind not in VALID_RUNTIME_EVENT_KINDS:
            raise ValueError(f"unsupported runtime event kind: {self.kind}")
        if self.kind == "approval_requested" and not isinstance(
            self.payload, ApprovalRequest
        ):
            raise ValueError(
                "approval_requested events require an ApprovalRequest payload"
            )
        if self.kind == "question_interrupted" and not isinstance(
            self.payload, UserQuestionRequest
        ):
            raise ValueError(
                "question_interrupted events require a UserQuestionRequest payload"
            )
        if self.kind == "resumed" and not isinstance(self.payload, ResumeRequest):
            raise ValueError("resumed events require a ResumeRequest payload")

    def as_dict(self) -> JSONObject:
        return {
            "kind": self.kind,
            "payload": self.payload.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "RuntimeEvent":
        kind = payload["kind"]
        event_payload = deep_copy_json_object(payload["payload"])
        if kind == "approval_requested":
            resolved_payload: EventPayload = ApprovalRequest.from_dict(event_payload)
        elif kind == "question_interrupted":
            resolved_payload = UserQuestionRequest.from_dict(event_payload)
        elif kind == "resumed":
            resolved_payload = ResumeRequest.from_dict(event_payload)
        else:
            raise ValueError(f"unsupported runtime event kind: {kind}")
        return cls(kind=kind, payload=resolved_payload)  # type: ignore[arg-type]

    @classmethod
    def approval_requested(cls, request: ApprovalRequest) -> "RuntimeEvent":
        return cls(kind="approval_requested", payload=request)

    @classmethod
    def question_interrupted(cls, request: UserQuestionRequest) -> "RuntimeEvent":
        return cls(kind="question_interrupted", payload=request)

    @classmethod
    def resumed(cls, request: ResumeRequest) -> "RuntimeEvent":
        return cls(kind="resumed", payload=request)


__all__ = [
    "ApprovalRequest",
    "ApprovalResponse",
    "ResumeRequest",
    "RuntimeEvent",
    "UserQuestionRequest",
    "UserQuestionResponse",
]

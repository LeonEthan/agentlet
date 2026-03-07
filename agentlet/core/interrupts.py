"""Core-owned interrupt, approval, and resume payload contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agentlet.core.types import (
    InterruptMetadata,
    InterruptOption,
    JSONObject,
    deep_copy_json_object,
)
from agentlet.tools.base import ApprovalCategory, VALID_APPROVAL_CATEGORIES

ApprovalDecisionValue = Literal["approved", "rejected"]
VALID_APPROVAL_DECISIONS = {"approved", "rejected"}

ResumeKind = Literal["approval", "question"]
VALID_RESUME_KINDS = {"approval", "question"}


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Structured approval prompt returned by the core loop."""

    request_id: str
    tool_name: str
    approval_category: ApprovalCategory
    prompt: str
    arguments: JSONObject = field(default_factory=dict)
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.tool_name:
            raise ValueError("tool_name must not be empty")
        if self.approval_category not in VALID_APPROVAL_CATEGORIES:
            raise ValueError(
                f"unsupported approval category: {self.approval_category}"
            )
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        object.__setattr__(self, "arguments", deep_copy_json_object(self.arguments))
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "approval_category": self.approval_category,
            "prompt": self.prompt,
        }
        if self.arguments:
            payload["arguments"] = deep_copy_json_object(self.arguments)
        if self.details:
            payload["details"] = deep_copy_json_object(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ApprovalRequest":
        return cls(
            request_id=str(payload["request_id"]),
            tool_name=str(payload["tool_name"]),
            approval_category=payload["approval_category"],  # type: ignore[arg-type]
            prompt=str(payload["prompt"]),
            arguments=deep_copy_json_object(payload.get("arguments", {})),
            details=deep_copy_json_object(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class ApprovalResponse:
    """Normalized user decision that resumes an approval gate."""

    request_id: str
    decision: ApprovalDecisionValue
    comment: str | None = None
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.decision not in VALID_APPROVAL_DECISIONS:
            raise ValueError(f"unsupported approval decision: {self.decision}")
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "request_id": self.request_id,
            "decision": self.decision,
        }
        if self.comment is not None:
            payload["comment"] = self.comment
        if self.details:
            payload["details"] = deep_copy_json_object(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ApprovalResponse":
        return cls(
            request_id=str(payload["request_id"]),
            decision=payload["decision"],  # type: ignore[arg-type]
            comment=(
                str(payload["comment"])
                if payload.get("comment") is not None
                else None
            ),
            details=deep_copy_json_object(payload.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class UserQuestionRequest:
    """A structured question interrupt that must be resumed explicitly."""

    request_id: str
    prompt: str
    options: tuple[InterruptOption, ...] = field(default_factory=tuple)
    allow_free_text: bool = False
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        object.__setattr__(self, "options", tuple(self.options))
        if not self.options and not self.allow_free_text:
            raise ValueError(
                "question requests must include options or allow_free_text=True"
            )
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "allow_free_text": self.allow_free_text,
        }
        if self.options:
            payload["options"] = [option.as_dict() for option in self.options]
        if self.details:
            payload["details"] = deep_copy_json_object(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "UserQuestionRequest":
        options_payload = payload.get("options", ())
        if not isinstance(options_payload, list | tuple):
            raise ValueError("question options must be a list")
        return cls(
            request_id=str(payload["request_id"]),
            prompt=str(payload["prompt"]),
            options=tuple(
                InterruptOption.from_dict(option_payload)
                for option_payload in options_payload
            ),
            allow_free_text=bool(payload.get("allow_free_text", False)),
            details=deep_copy_json_object(payload.get("details", {})),
        )

    @classmethod
    def from_interrupt(cls, interrupt: InterruptMetadata) -> "UserQuestionRequest":
        if interrupt.kind != "question":
            raise ValueError("only question interrupts can become question requests")
        if interrupt.request_id is None:
            raise ValueError("question interrupts must include request_id")
        return cls(
            request_id=interrupt.request_id,
            prompt=interrupt.prompt,
            options=interrupt.options,
            allow_free_text=interrupt.allow_free_text,
            details=interrupt.details,
        )

    def validate_response(self, response: "UserQuestionResponse") -> None:
        if response.request_id != self.request_id:
            raise ValueError("question response request_id does not match request")
        if response.selected_option is not None:
            allowed_values = {option.value for option in self.options}
            if response.selected_option not in allowed_values:
                raise ValueError("question response selected_option is not valid")
        if response.free_text is not None and not self.allow_free_text:
            raise ValueError("question response free_text is not allowed")


@dataclass(frozen=True, slots=True)
class UserQuestionResponse:
    """Structured answer supplied when resuming a question interrupt."""

    request_id: str
    selected_option: str | None = None
    free_text: str | None = None
    details: JSONObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.selected_option is None and self.free_text is None:
            raise ValueError(
                "question responses must include selected_option or free_text"
            )
        if self.selected_option is not None and self.free_text is not None:
            raise ValueError(
                "question responses must include selected_option or free_text, not both"
            )
        if self.selected_option is not None and not self.selected_option.strip():
            raise ValueError("selected_option must not be empty")
        if self.free_text is not None and not self.free_text.strip():
            raise ValueError("free_text must not be empty")
        object.__setattr__(self, "details", deep_copy_json_object(self.details))

    def as_dict(self) -> JSONObject:
        payload: JSONObject = {"request_id": self.request_id}
        if self.selected_option is not None:
            payload["selected_option"] = self.selected_option
        if self.free_text is not None:
            payload["free_text"] = self.free_text
        if self.details:
            payload["details"] = deep_copy_json_object(self.details)
        return payload

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "UserQuestionResponse":
        return cls(
            request_id=str(payload["request_id"]),
            selected_option=(
                str(payload["selected_option"])
                if payload.get("selected_option") is not None
                else None
            ),
            free_text=(
                str(payload["free_text"])
                if payload.get("free_text") is not None
                else None
            ),
            details=deep_copy_json_object(payload.get("details", {})),
        )


ResumePayload = ApprovalResponse | UserQuestionResponse


@dataclass(frozen=True, slots=True)
class ResumeRequest:
    """Typed resume input passed back into the loop after a pause."""

    kind: ResumeKind
    payload: ResumePayload

    def __post_init__(self) -> None:
        if self.kind not in VALID_RESUME_KINDS:
            raise ValueError(f"unsupported resume kind: {self.kind}")
        if self.kind == "approval" and not isinstance(self.payload, ApprovalResponse):
            raise ValueError("approval resumes require an ApprovalResponse payload")
        if self.kind == "question" and not isinstance(
            self.payload, UserQuestionResponse
        ):
            raise ValueError(
                "question resumes require a UserQuestionResponse payload"
            )

    def as_dict(self) -> JSONObject:
        return {
            "kind": self.kind,
            "payload": self.payload.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: JSONObject) -> "ResumeRequest":
        kind = payload["kind"]
        if kind == "approval":
            resume_payload: ResumePayload = ApprovalResponse.from_dict(
                deep_copy_json_object(payload["payload"])
            )
        elif kind == "question":
            resume_payload = UserQuestionResponse.from_dict(
                deep_copy_json_object(payload["payload"])
            )
        else:
            raise ValueError(f"unsupported resume kind: {kind}")
        return cls(kind=kind, payload=resume_payload)  # type: ignore[arg-type]

    @classmethod
    def from_approval_response(cls, response: ApprovalResponse) -> "ResumeRequest":
        return cls(kind="approval", payload=response)

    @classmethod
    def from_question_response(
        cls, response: UserQuestionResponse
    ) -> "ResumeRequest":
        return cls(kind="question", payload=response)


__all__ = [
    "ApprovalDecisionValue",
    "ApprovalRequest",
    "ApprovalResponse",
    "ResumeKind",
    "ResumePayload",
    "ResumeRequest",
    "UserQuestionRequest",
    "UserQuestionResponse",
]

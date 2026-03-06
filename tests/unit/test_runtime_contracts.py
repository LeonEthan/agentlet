from __future__ import annotations

import pytest

from agentlet.core.types import InterruptMetadata, InterruptOption
from agentlet.runtime.events import (
    ApprovalRequest,
    ApprovalResponse,
    ResumeRequest,
    RuntimeEvent,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.runtime.user_io import UserIO


def test_approval_runtime_event_schema_is_stable() -> None:
    request = ApprovalRequest(
        request_id="approval_1",
        tool_name="Bash",
        approval_category="exec",
        prompt="Allow `pytest` to run?",
        arguments={"command": "pytest", "timeout_seconds": 30},
        details={"cwd": "/repo"},
    )

    payload = RuntimeEvent.approval_requested(request).as_dict()

    assert payload == {
        "kind": "approval_requested",
        "payload": {
            "request_id": "approval_1",
            "tool_name": "Bash",
            "approval_category": "exec",
            "prompt": "Allow `pytest` to run?",
            "arguments": {"command": "pytest", "timeout_seconds": 30},
            "details": {"cwd": "/repo"},
        },
    }
    assert RuntimeEvent.from_dict(payload) == RuntimeEvent.approval_requested(request)


def test_question_interrupt_event_schema_is_stable() -> None:
    interrupt = InterruptMetadata(
        kind="question",
        prompt="Which file should I inspect?",
        request_id="question_1",
        options=[
            InterruptOption(value="readme", label="README.md"),
            InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
        ],
        allow_free_text=True,
        details={"source_tool": "AskUserQuestion"},
    )

    request = UserQuestionRequest.from_interrupt(interrupt)
    payload = RuntimeEvent.question_interrupted(request).as_dict()

    assert payload == {
        "kind": "question_interrupted",
        "payload": {
            "request_id": "question_1",
            "prompt": "Which file should I inspect?",
            "allow_free_text": True,
            "options": [
                {"value": "readme", "label": "README.md"},
                {"value": "arch", "label": "docs/ARCHITECTURE.md"},
            ],
            "details": {"source_tool": "AskUserQuestion"},
        },
    }
    assert RuntimeEvent.from_dict(payload) == RuntimeEvent.question_interrupted(request)


def test_question_interrupt_contract_requires_request_id() -> None:
    interrupt = InterruptMetadata(
        kind="question",
        prompt="Need clarification",
    )

    with pytest.raises(ValueError, match="must include request_id"):
        UserQuestionRequest.from_interrupt(interrupt)


def test_question_interrupt_contract_requires_options_or_free_text() -> None:
    with pytest.raises(
        ValueError,
        match="must include options or allow_free_text=True",
    ):
        UserQuestionRequest(
            request_id="question_1",
            prompt="Need clarification",
        )


def test_resume_request_schema_is_stable_and_validates_payload_kind() -> None:
    response_details = {"resume": {"channel": "cli"}}
    response = UserQuestionResponse(
        request_id="question_1",
        selected_option="readme",
        details=response_details,
    )
    resume = ResumeRequest.from_question_response(response)
    response_details["resume"]["channel"] = "changed"

    assert resume.as_dict() == {
        "kind": "question",
        "payload": {
            "request_id": "question_1",
            "selected_option": "readme",
            "details": {"resume": {"channel": "cli"}},
        },
    }
    assert ResumeRequest.from_dict(resume.as_dict()) == resume

    with pytest.raises(
        ValueError, match="question responses must include selected_option or free_text"
    ):
        UserQuestionResponse(request_id="question_1")

    with pytest.raises(ValueError, match="approval resumes require"):
        ResumeRequest(
            kind="approval",
            payload=UserQuestionResponse(
                request_id="question_1",
                selected_option="readme",
            ),
        )


def test_approval_resume_event_round_trips() -> None:
    response = ApprovalResponse(
        request_id="approval_1",
        decision="approved",
        comment="Looks safe.",
        details={"reviewed_by": "user"},
    )
    event = RuntimeEvent.resumed(ResumeRequest.from_approval_response(response))

    assert event.as_dict() == {
        "kind": "resumed",
        "payload": {
            "kind": "approval",
            "payload": {
                "request_id": "approval_1",
                "decision": "approved",
                "comment": "Looks safe.",
                "details": {"reviewed_by": "user"},
            },
        },
    }
    assert RuntimeEvent.from_dict(event.as_dict()) == event


def test_user_io_protocol_is_runtime_checkable() -> None:
    class FakeUserIO:
        def emit_event(self, event: RuntimeEvent) -> None:
            self.last_event = event

        def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
            return ApprovalResponse(
                request_id=request.request_id,
                decision="approved",
            )

        def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
            self.last_question = request

        def resolve_question_interrupt(
            self,
            request: UserQuestionRequest,
        ) -> UserQuestionResponse:
            return UserQuestionResponse(
                request_id=request.request_id,
                selected_option="readme",
            )

    assert isinstance(FakeUserIO(), UserIO)

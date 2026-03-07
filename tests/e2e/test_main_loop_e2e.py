from __future__ import annotations

from typing import Any

from agentlet.core.interrupts import (
    ApprovalRequest,
    ApprovalResponse,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.core.loop import CompletedTurn
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptOption
from agentlet.llm.schemas import ModelRequest, ModelResponse
from agentlet.runtime.app import build_runtime_app
from agentlet.runtime.events import RuntimeEvent
from agentlet.tools.fs.write import WriteTool
from agentlet.tools.interaction.ask_user_question import AskUserQuestionTool
from agentlet.tools.registry import ToolRegistry


class FakeUserIO:
    def __init__(
        self,
        *,
        approval_decisions: list[str] | None = None,
        question_responses: list[UserQuestionResponse] | None = None,
    ) -> None:
        self.approval_decisions = list(approval_decisions or [])
        self.question_responses = list(question_responses or [])
        self.events: list[RuntimeEvent] = []
        self.approval_requests: list[ApprovalRequest] = []
        self.question_requests: list[UserQuestionRequest] = []

    def emit_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.approval_requests.append(request)
        decision = (
            self.approval_decisions.pop(0)
            if self.approval_decisions
            else "approved"
        )
        return ApprovalResponse(
            request_id=request.request_id,
            decision=decision,  # type: ignore[arg-type]
        )

    def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
        self.question_requests.append(request)

    def resolve_question_interrupt(
        self,
        request: UserQuestionRequest,
    ) -> UserQuestionResponse:
        if self.question_responses:
            return self.question_responses.pop(0)
        return UserQuestionResponse(
            request_id=request.request_id,
            selected_option=request.options[0].value,
        )


class FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("no fake model responses remaining")
        return self._responses.pop(0)


def _message_history(app: Any) -> list[Message]:
    return [
        Message.from_dict(record.payload)
        for record in app.loop.session_store.load()
        if record.kind == "message"
    ]


def test_end_to_end_local_coding_path_creates_documented_persistence_files(
    tmp_path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("You are agentlet.\n", encoding="utf-8")
    user_io = FakeUserIO(approval_decisions=["approved"])
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I should write the file.",
                    tool_calls=(
                        ToolCall(
                            id="call_write",
                            name="Write",
                            arguments={"path": "notes.md", "content": "hello"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="Proceeding with the approved write.",
                    tool_calls=(
                        ToolCall(
                            id="call_write",
                            name="Write",
                            arguments={"path": "notes.md", "content": "hello"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="notes.md is ready."),
                finish_reason="stop",
            ),
        ]
    )

    app = build_runtime_app(
        model=model,
        user_io=user_io,
        workspace_root=tmp_path,
        registry=ToolRegistry([WriteTool(tmp_path)]),
    )

    outcome = app.run_turn(current_task="Create notes.md")

    assert isinstance(outcome, CompletedTurn)
    assert (tmp_path / ".agentlet" / "session.jsonl").exists()
    assert (tmp_path / ".agentlet" / "memory.md").exists()
    assert app.system_instructions == "You are agentlet.\n"
    assert (tmp_path / "notes.md").read_text(encoding="utf-8") == "hello"
    assert [event.kind for event in user_io.events] == [
        "approval_requested",
        "resumed",
    ]
    assert _message_history(app)[-1] == Message(
        role="assistant",
        content="notes.md is ready.",
    )


def test_end_to_end_approval_refusal_path_keeps_workspace_unchanged(tmp_path) -> None:
    user_io = FakeUserIO(approval_decisions=["rejected"])
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I should write the file.",
                    tool_calls=(
                        ToolCall(
                            id="call_write",
                            name="Write",
                            arguments={"path": "notes.md", "content": "hello"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I would write the file.",
                    tool_calls=(
                        ToolCall(
                            id="call_write",
                            name="Write",
                            arguments={"path": "notes.md", "content": "hello"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="Skipping the write."),
                finish_reason="stop",
            ),
        ]
    )

    app = build_runtime_app(
        model=model,
        user_io=user_io,
        workspace_root=tmp_path,
        registry=ToolRegistry([WriteTool(tmp_path)]),
    )

    outcome = app.run_turn(current_task="Create notes.md")

    assert isinstance(outcome, CompletedTurn)
    assert not (tmp_path / "notes.md").exists()
    assert _message_history(app)[-2:] == [
        Message(
            role="tool",
            name="Write",
            content="Tool `Write` was not executed because approval was rejected.",
            tool_call_id="call_write",
            metadata={
                "tool_name": "Write",
                "is_error": True,
                "result": {
                    "tool_name": "Write",
                    "error_type": "ApprovalRejected",
                    "request_id": user_io.approval_requests[0].request_id,
                },
            },
        ),
        Message(role="assistant", content="Skipping the write."),
    ]
    assert [event.kind for event in user_io.events] == [
        "approval_requested",
        "resumed",
    ]


def test_end_to_end_interrupt_path_persists_resume_context_and_finishes(
    tmp_path,
) -> None:
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I need clarification.",
                    tool_calls=(
                        ToolCall(
                            id="call_question",
                            name="AskUserQuestion",
                            arguments={
                                "prompt": "Which file should I edit?",
                                "request_id": "question_1",
                                "options": [
                                    {"value": "readme", "label": "README.md"},
                                    {"value": "arch", "label": "docs/ARCHITECTURE.md"},
                                ],
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="Use README.md."),
                finish_reason="stop",
            ),
        ]
    )
    user_io = FakeUserIO(
        question_responses=[
            UserQuestionResponse(
                request_id="question_1",
                selected_option="readme",
            )
        ]
    )

    app = build_runtime_app(
        model=model,
        user_io=user_io,
        workspace_root=tmp_path,
        registry=ToolRegistry([AskUserQuestionTool()]),
    )

    outcome = app.run_turn(current_task="Pick a file to edit.")

    assert isinstance(outcome, CompletedTurn)
    assert user_io.question_requests == [
        UserQuestionRequest(
            request_id="question_1",
            prompt="Which file should I edit?",
            options=(
                InterruptOption(value="readme", label="README.md"),
                InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
            ),
            details={"source_tool": "AskUserQuestion"},
        )
    ]
    assert [event.kind for event in user_io.events] == [
        "question_interrupted",
        "resumed",
    ]
    assert _message_history(app)[-3:] == [
        Message(
            role="tool",
            name="AskUserQuestion",
            content="Awaiting user response.",
            tool_call_id="call_question",
            metadata={
                "tool_name": "AskUserQuestion",
                "is_error": False,
                "result": {
                    "interrupt": {
                        "kind": "question",
                        "prompt": "Which file should I edit?",
                        "request_id": "question_1",
                        "options": [
                            {"value": "readme", "label": "README.md"},
                            {"value": "arch", "label": "docs/ARCHITECTURE.md"},
                        ],
                        "allow_free_text": False,
                        "details": {"source_tool": "AskUserQuestion"},
                    }
                },
            },
        ),
        Message(
            role="user",
            content=(
                "Interrupt resume context:\n"
                "{\n"
                '  "kind": "question",\n'
                '  "request_id": "question_1",\n'
                '  "selected_option": "readme"\n'
                "}"
            ),
        ),
        Message(role="assistant", content="Use README.md."),
    ]

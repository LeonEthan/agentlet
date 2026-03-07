from __future__ import annotations

from agentlet.core.approvals import ApprovalPolicy
from agentlet.core.interrupts import (
    ApprovalRequest,
    ApprovalResponse,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.core.loop import CompletedTurn
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, InterruptOption
from agentlet.llm.schemas import ModelRequest, ModelResponse
from agentlet.runtime.app import build_runtime_app
from agentlet.runtime.events import RuntimeEvent
from agentlet.tools.base import ToolDefinition, ToolResult
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
        self.question_requests: list[UserQuestionRequest] = []
        self.approval_requests: list[ApprovalRequest] = []

    def emit_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.approval_requests.append(request)
        decision = self.approval_decisions.pop(0) if self.approval_decisions else "approved"
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
            selected_option="default",
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


class FakeTool:
    def __init__(self, definition: ToolDefinition, result: ToolResult) -> None:
        self.definition = definition
        self.result = result
        self.executed_arguments: list[dict[str, object]] = []

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        self.executed_arguments.append(arguments)
        return self.result


class FakeRegistry:
    def __init__(self, tools: list[FakeTool]) -> None:
        self._tools = {tool.definition.name: tool for tool in tools}

    def get(self, name: str) -> FakeTool | None:
        return self._tools.get(name)

    def definition(self, name: str) -> ToolDefinition:
        return self._tools[name].definition

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools.values())


def test_build_runtime_app_assembles_default_components_and_runs_minimal_session(
    tmp_path,
) -> None:
    (tmp_path / "AGENTS.md").write_text("You are agentlet.\n", encoding="utf-8")
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(role="assistant", content="Ready."),
                finish_reason="stop",
            )
        ]
    )

    app = build_runtime_app(
        model=model,
        user_io=FakeUserIO(),
        workspace_root=tmp_path,
    )

    assert app.system_instructions == "You are agentlet.\n"
    assert app.loop.session_store.path == (tmp_path / ".agentlet" / "session.jsonl")
    assert app.loop.memory_store is not None
    assert app.loop.memory_store.path == (tmp_path / ".agentlet" / "memory.md")
    assert app.loop.session_store.path.exists()
    assert app.loop.session_store.path.read_text(encoding="utf-8") == ""
    assert app.loop.memory_store.path.exists()
    assert app.loop.memory_store.path.read_text(encoding="utf-8") == ""
    assert tuple(
        definition.name for definition in app.loop.registry.definitions()
    ) == (
        "Read",
        "Write",
        "Edit",
        "Bash",
        "Glob",
        "Grep",
        "WebSearch",
        "WebFetch",
        "AskUserQuestion",
    )
    outcome = app.run_turn(current_task="Say ready.")
    assert isinstance(outcome, CompletedTurn)
    assert outcome.message.content == "Ready."


def test_build_runtime_app_preserves_explicit_empty_registry(tmp_path) -> None:
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(role="assistant", content="No tools."),
                finish_reason="stop",
            )
        ]
    )
    empty_registry = ToolRegistry()

    app = build_runtime_app(
        model=model,
        user_io=FakeUserIO(),
        workspace_root=tmp_path,
        registry=empty_registry,
    )

    assert app.loop.registry is empty_registry
    assert app.loop.registry.definitions() == ()


def test_runtime_app_auto_resolves_approval_requests(tmp_path) -> None:
    write_tool = FakeTool(
        ToolDefinition(
            name="Write",
            description="Write a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            approval_category="mutating",
        ),
        ToolResult(output="Created file: notes.md"),
    )
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
                message=Message(role="assistant", content="Done."),
                finish_reason="stop",
            ),
        ]
    )
    user_io = FakeUserIO(approval_decisions=["approved"])

    app = build_runtime_app(
        model=model,
        user_io=user_io,
        workspace_root=tmp_path,
        registry=FakeRegistry([write_tool]),
    )

    outcome = app.run_turn(current_task="Create notes.md")

    assert isinstance(outcome, CompletedTurn)
    assert write_tool.executed_arguments == [{"path": "notes.md", "content": "hello"}]
    assert [event.kind for event in user_io.events] == [
        "approval_requested",
        "resumed",
    ]
    stored_messages = [
        Message.from_dict(record.payload)
        for record in app.loop.session_store.load()
        if record.kind == "message"
    ]
    assert stored_messages[-1] == Message(role="assistant", content="Done.")


def test_runtime_app_resumes_question_interrupts_through_user_io(tmp_path) -> None:
    ask_tool = FakeTool(
        ToolDefinition(
            name="AskUserQuestion",
            description="Ask the user a clarifying question.",
            input_schema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
                "additionalProperties": False,
            },
            approval_category="external_or_interrupt",
        ),
        ToolResult.interrupt_result(
            output="Need clarification.",
            interrupt=InterruptMetadata(
                kind="question",
                prompt="Which file should I edit?",
                request_id="question_1",
                options=(
                    InterruptOption(value="readme", label="README.md"),
                    InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
                ),
            ),
        ),
    )
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
                            arguments={"prompt": "Which file should I edit?"},
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
        registry=FakeRegistry([ask_tool]),
        approval_policy=ApprovalPolicy(
            {"external_or_interrupt": "allow"}
        ),
    )

    outcome = app.run_turn(current_task="Continue.")

    assert isinstance(outcome, CompletedTurn)
    assert user_io.question_requests == [
        UserQuestionRequest(
            request_id="question_1",
            prompt="Which file should I edit?",
            options=(
                InterruptOption(value="readme", label="README.md"),
                InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
            ),
        )
    ]
    assert [event.kind for event in user_io.events] == [
        "question_interrupted",
        "resumed",
    ]
    assert outcome.message.content == "Use README.md."


def test_runtime_app_resumes_question_interrupts_with_free_text(tmp_path) -> None:
    ask_tool = FakeTool(
        ToolDefinition(
            name="AskUserQuestion",
            description="Ask the user a clarifying question.",
            input_schema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
                "additionalProperties": False,
            },
            approval_category="external_or_interrupt",
        ),
        ToolResult.interrupt_result(
            output="Need clarification.",
            interrupt=InterruptMetadata(
                kind="question",
                prompt="Which file should I edit?",
                request_id="question_1",
                allow_free_text=True,
            ),
        ),
    )
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
                            arguments={"prompt": "Which file should I edit?"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="Use notes.md."),
                finish_reason="stop",
            ),
        ]
    )
    user_io = FakeUserIO(
        question_responses=[
            UserQuestionResponse(
                request_id="question_1",
                free_text="notes.md",
            )
        ]
    )

    app = build_runtime_app(
        model=model,
        user_io=user_io,
        workspace_root=tmp_path,
        registry=FakeRegistry([ask_tool]),
        approval_policy=ApprovalPolicy(
            {"external_or_interrupt": "allow"}
        ),
    )

    outcome = app.run_turn(current_task="Continue.")

    assert isinstance(outcome, CompletedTurn)
    assert [event.kind for event in user_io.events] == [
        "question_interrupted",
        "resumed",
    ]
    assert outcome.message.content == "Use notes.md."


def test_default_runtime_registry_supports_question_interrupt_flow(tmp_path) -> None:
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
    )

    outcome = app.run_turn(current_task="Continue.")

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
    assert outcome.message.content == "Use README.md."

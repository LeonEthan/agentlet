from __future__ import annotations

from dataclasses import asdict

import pytest

from agentlet.core.approvals import ApprovalPolicy
from agentlet.core.context import CurrentTaskState
from agentlet.core.loop import (
    AgentLoop,
    ApprovalRequiredTurn,
    CompletedTurn,
    InterruptedTurn,
)
from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, InterruptOption
from agentlet.llm.schemas import ModelRequest, ModelResponse
from agentlet.memory import SessionRecord
from agentlet.runtime.events import ResumeRequest, UserQuestionResponse
from agentlet.tools.base import ToolDefinition, ToolResult


class FakeSessionStore:
    def __init__(self, records: list[SessionRecord] | None = None) -> None:
        self.records = list(records or [])
        self.append_calls: list[list[SessionRecord]] = []
        self.load_calls = 0

    def load(self, *, skip_malformed: bool = False) -> list[SessionRecord]:
        assert skip_malformed is False
        self.load_calls += 1
        return list(self.records)

    def append_many(
        self,
        records: list[SessionRecord | dict[str, object]],
    ) -> list[SessionRecord]:
        normalized = [
            record if isinstance(record, SessionRecord) else SessionRecord.from_dict(record)
            for record in records
        ]
        self.append_calls.append(normalized)
        self.records.extend(normalized)
        return normalized


class FakeMemoryStore:
    def __init__(self, content: str = "") -> None:
        self.content = content
        self.read_calls = 0

    def read(self) -> str:
        self.read_calls += 1
        return self.content


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
    def __init__(
        self,
        definition: ToolDefinition,
        *,
        result: ToolResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.definition = definition
        self.result = result or ToolResult(output=f"{definition.name} ok")
        self.error = error
        self.executed_arguments: list[dict[str, object]] = []

    def execute(self, arguments: dict[str, object]) -> ToolResult:
        self.executed_arguments.append(arguments)
        if self.error is not None:
            raise self.error
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


def _definition(name: str, approval_category: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"{name} description",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
        approval_category=approval_category,  # type: ignore[arg-type]
    )


def _message_records(store: FakeSessionStore) -> tuple[Message, ...]:
    return tuple(Message.from_dict(record.payload) for record in store.records)


def test_agent_loop_completes_without_tools_and_persists_turn() -> None:
    session_store = FakeSessionStore(
        [
            SessionRecord(
                record_id="message_1",
                kind="message",
                payload=asdict(Message(role="user", content="Previous task")),
            ),
            SessionRecord(
                record_id="message_2",
                kind="message",
                payload=asdict(Message(role="assistant", content="Previous answer")),
            ),
        ]
    )
    memory_store = FakeMemoryStore("# Memory\n- Prefer tests.")
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(role="assistant", content="Here is the answer."),
                finish_reason="stop",
            )
        ]
    )
    registry = FakeRegistry([])

    outcome = AgentLoop(
        model=model,
        registry=registry,
        session_store=session_store,
        memory_store=memory_store,
    ).run(
        current_task=CurrentTaskState(
            task="Summarize the architecture.",
            details={"cwd": "/repo"},
        ),
        system_instructions="You are agentlet.",
    )

    assert isinstance(outcome, CompletedTurn)
    assert outcome.message.content == "Here is the answer."
    assert memory_store.read_calls == 1
    assert session_store.load_calls == 1
    assert len(model.requests) == 1
    assert model.requests[0].messages == (
        Message(role="system", content="You are agentlet."),
        Message(role="system", content="Durable memory:\n# Memory\n- Prefer tests."),
        Message(role="user", content="Previous task"),
        Message(role="assistant", content="Previous answer"),
        Message(
            role="user",
            content=(
                "Summarize the architecture.\n\n"
                'Task state:\n{\n  "cwd": "/repo"\n}'
            ),
        ),
    )
    assert _message_records(session_store)[-2:] == (
        Message(
            role="user",
            content=(
                "Summarize the architecture.\n\n"
                'Task state:\n{\n  "cwd": "/repo"\n}'
            ),
        ),
        Message(role="assistant", content="Here is the answer."),
    )


def test_agent_loop_executes_tool_calls_and_continues_until_final_response() -> None:
    read_tool = FakeTool(
        _definition("Read", "read_only"),
        result=ToolResult(output="# README"),
    )
    grep_tool = FakeTool(
        _definition("Grep", "read_only"),
        result=ToolResult(output="README.md:1:agentlet"),
    )
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I will inspect the repo.",
                    tool_calls=(
                        ToolCall(
                            id="call_read",
                            name="Read",
                            arguments={"path": "README.md"},
                        ),
                        ToolCall(
                            id="call_grep",
                            name="Grep",
                            arguments={"path": "README.md", "pattern": "agentlet"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="The repo builds agentlet."),
                finish_reason="stop",
            ),
        ]
    )
    session_store = FakeSessionStore()

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([read_tool, grep_tool]),
        session_store=session_store,
    ).run(current_task="Inspect the repo.")

    assert isinstance(outcome, CompletedTurn)
    assert outcome.message.content == "The repo builds agentlet."
    assert read_tool.executed_arguments == [{"path": "README.md"}]
    assert grep_tool.executed_arguments == [
        {"path": "README.md", "pattern": "agentlet"}
    ]
    assert len(model.requests) == 2
    assert model.requests[1].messages[-3:] == (
        Message(
            role="assistant",
            content="I will inspect the repo.",
            tool_calls=(
                ToolCall(
                    id="call_read",
                    name="Read",
                    arguments={"path": "README.md"},
                ),
                ToolCall(
                    id="call_grep",
                    name="Grep",
                    arguments={"path": "README.md", "pattern": "agentlet"},
                ),
            ),
        ),
        Message(
            role="tool",
            name="Read",
            content="# README",
            tool_call_id="call_read",
            metadata={"tool_name": "Read", "is_error": False},
        ),
        Message(
            role="tool",
            name="Grep",
            content="README.md:1:agentlet",
            tool_call_id="call_grep",
            metadata={"tool_name": "Grep", "is_error": False},
        ),
    )
    assert _message_records(session_store) == (
        Message(role="user", content="Inspect the repo."),
        Message(
            role="assistant",
            content="I will inspect the repo.",
            tool_calls=(
                ToolCall(
                    id="call_read",
                    name="Read",
                    arguments={"path": "README.md"},
                ),
                ToolCall(
                    id="call_grep",
                    name="Grep",
                    arguments={"path": "README.md", "pattern": "agentlet"},
                ),
            ),
        ),
        Message(
            role="tool",
            name="Read",
            content="# README",
            tool_call_id="call_read",
            metadata={"tool_name": "Read", "is_error": False},
        ),
        Message(
            role="tool",
            name="Grep",
            content="README.md:1:agentlet",
            tool_call_id="call_grep",
            metadata={"tool_name": "Grep", "is_error": False},
        ),
        Message(role="assistant", content="The repo builds agentlet."),
    )


def test_agent_loop_surfaces_tool_failures_back_to_model() -> None:
    read_tool = FakeTool(
        _definition("Read", "read_only"),
        error=RuntimeError("boom"),
    )
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="I will inspect the file.",
                    tool_calls=(
                        ToolCall(
                            id="call_1",
                            name="Read",
                            arguments={"path": "README.md"},
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(role="assistant", content="The read failed."),
                finish_reason="stop",
            ),
        ]
    )

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([read_tool]),
        session_store=FakeSessionStore(),
    ).run(current_task="Inspect README.md")

    assert isinstance(outcome, CompletedTurn)
    tool_message = model.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.metadata["is_error"] is True
    assert tool_message.content == "Tool `Read` failed: boom"


def test_agent_loop_rejects_invalid_tool_arguments_before_approval_or_execution() -> None:
    write_tool = FakeTool(
        ToolDefinition(
            name="Write",
            description="Write description",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            approval_category="mutating",
        )
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
                            arguments={
                                "path": "notes.md",
                                "content": "hello",
                                "mode": "append",
                            },
                        ),
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(
                message=Message(
                    role="assistant",
                    content="The write arguments were invalid.",
                ),
                finish_reason="stop",
            ),
        ]
    )

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([write_tool]),
        session_store=FakeSessionStore(),
    ).run(current_task="Create notes.md")

    assert isinstance(outcome, CompletedTurn)
    assert write_tool.executed_arguments == []
    tool_message = model.requests[1].messages[-1]
    assert tool_message.role == "tool"
    assert tool_message.metadata["is_error"] is True
    assert tool_message.content == (
        "Tool `Write` received invalid arguments: unexpected arguments: mode"
    )


def test_agent_loop_returns_structured_approval_outcome_before_execution() -> None:
    write_tool = FakeTool(_definition("Write", "mutating"))
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
            )
        ]
    )
    session_store = FakeSessionStore()

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([write_tool]),
        session_store=session_store,
    ).run(current_task="Create notes.md")

    assert isinstance(outcome, ApprovalRequiredTurn)
    assert outcome.request.request_id == "approval:call_write"
    assert outcome.request.tool_name == "Write"
    assert outcome.request.arguments == {"path": "notes.md", "content": "hello"}
    assert outcome.request.details == {
        "tool_call_id": "call_write",
        "reason": "mutating tools require runtime approval before execution.",
    }
    assert write_tool.executed_arguments == []
    assert _message_records(session_store) == (
        Message(role="user", content="Create notes.md"),
        Message(
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
    )


def test_agent_loop_returns_structured_interrupt_outcome_and_persists_partial_turn() -> None:
    ask_tool = FakeTool(
        _definition("AskUserQuestion", "external_or_interrupt"),
        result=ToolResult.interrupt_result(
            output="Need clarification before editing.",
            interrupt=InterruptMetadata(
                kind="question",
                prompt="Which file should I edit?",
                request_id="question_1",
                options=(
                    InterruptOption(value="readme", label="README.md"),
                    InterruptOption(value="arch", label="docs/ARCHITECTURE.md"),
                ),
                allow_free_text=True,
            ),
            metadata={"source_tool": "AskUserQuestion"},
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
            )
        ]
    )

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([ask_tool]),
        session_store=FakeSessionStore(),
        approval_policy=ApprovalPolicy({"external_or_interrupt": "allow"}),
    ).run(current_task="Choose a file to edit.")

    assert isinstance(outcome, InterruptedTurn)
    assert outcome.interrupt.prompt == "Which file should I edit?"
    assert outcome.tool_message.metadata == {
        "tool_name": "AskUserQuestion",
        "is_error": False,
        "result": {
            "source_tool": "AskUserQuestion",
            "interrupt": {
                "kind": "question",
                "prompt": "Which file should I edit?",
                "request_id": "question_1",
                "options": [
                    {"value": "readme", "label": "README.md"},
                    {"value": "arch", "label": "docs/ARCHITECTURE.md"},
                ],
                "allow_free_text": True,
            },
        },
    }


def test_agent_loop_includes_resume_context_in_the_next_model_request() -> None:
    session_store = FakeSessionStore(
        [
            SessionRecord(
                record_id="message_1",
                kind="message",
                payload=asdict(Message(role="user", content="Pick a file.")),
            ),
            SessionRecord(
                record_id="message_2",
                kind="message",
                payload=asdict(
                    Message(
                        role="assistant",
                        content="I need clarification.",
                        tool_calls=(
                            ToolCall(
                                id="call_question",
                                name="AskUserQuestion",
                                arguments={"prompt": "Which file should I edit?"},
                            ),
                        ),
                    )
                ),
            ),
            SessionRecord(
                record_id="message_3",
                kind="message",
                payload=asdict(
                    Message(
                        role="tool",
                        name="AskUserQuestion",
                        content="Need clarification before editing.",
                        tool_call_id="call_question",
                        metadata={
                            "tool_name": "AskUserQuestion",
                            "is_error": False,
                            "result": {
                                "interrupt": {
                                    "kind": "question",
                                    "prompt": "Which file should I edit?",
                                    "request_id": "question_1",
                                }
                            },
                        },
                    )
                ),
            ),
        ]
    )
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(role="assistant", content="I will edit README.md."),
                finish_reason="stop",
            )
        ]
    )

    outcome = AgentLoop(
        model=model,
        registry=FakeRegistry([]),
        session_store=session_store,
    ).run(
        current_task="Continue the task.",
        resume=ResumeRequest.from_question_response(
            UserQuestionResponse(
                request_id="question_1",
                selected_option="readme",
            )
        ),
    )

    assert isinstance(outcome, CompletedTurn)
    assert model.requests[0].messages[-2:] == (
        Message(
            role="user",
            content=(
                "Interrupt resume context:\n"
                '{\n  "kind": "question",\n'
                '  "request_id": "question_1",\n'
                '  "selected_option": "readme"\n}'
            ),
        ),
        Message(role="user", content="Continue the task."),
    )


def test_agent_loop_rejects_non_terminal_assistant_response_without_tool_calls() -> None:
    session_store = FakeSessionStore()
    model = FakeModelClient(
        [
            ModelResponse(
                message=Message(role="assistant", content="Partial answer"),
                finish_reason="length",
            )
        ]
    )

    with pytest.raises(
        RuntimeError,
        match="model returned a non-terminal assistant response without tool calls: length",
    ):
        AgentLoop(
            model=model,
            registry=FakeRegistry([]),
            session_store=session_store,
        ).run(current_task="Answer the question.")

    assert session_store.append_calls == []

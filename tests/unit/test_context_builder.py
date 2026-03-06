from __future__ import annotations

import pytest

from agentlet.core.context import (
    ContextBuilder,
    CurrentTaskState,
    PendingInterruptContext,
)
from agentlet.core.messages import Message, ToolCall
from agentlet.memory import SessionRecord


def test_context_builder_returns_empty_tuple_for_empty_inputs() -> None:
    builder = ContextBuilder()

    assert builder.build() == ()


def test_context_builder_builds_context_without_memory() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        system_instructions="You are agentlet.",
        session_history=[
            SessionRecord(
                record_id="user_1",
                kind="message",
                payload={"role": "user", "content": "Read README.md"},
            ),
            Message(
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
            {
                "role": "tool",
                "content": "# README",
                "tool_call_id": "call_1",
            },
        ],
        current_task="Summarize the architecture decisions.",
    )

    assert messages == (
        Message(role="system", content="You are agentlet."),
        Message(role="user", content="Read README.md"),
        Message(
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
        Message(role="tool", content="# README", tool_call_id="call_1"),
        Message(role="user", content="Summarize the architecture decisions."),
    )


def test_context_builder_builds_context_without_history() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        system_instructions="You are agentlet.",
        durable_memory="# Memory\n- Prefer tests.",
        current_task=CurrentTaskState(
            task="Summarize the architecture decisions.",
            details={"cwd": "/repo", "resume": {"kind": "fresh"}},
        ),
    )

    assert messages == (
        Message(role="system", content="You are agentlet."),
        Message(role="system", content="Durable memory:\n# Memory\n- Prefer tests."),
        Message(
            role="user",
            content=(
                "Summarize the architecture decisions.\n\n"
                "Task state:\n"
                '{\n  "cwd": "/repo",\n  "resume": {\n    "kind": "fresh"\n  }\n}'
            ),
        ),
    )


def test_context_builder_orders_memory_history_interrupt_and_task() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        system_instructions="You are agentlet.",
        durable_memory="# Memory\n- Prefer tests.",
        session_history=[Message(role="user", content="Read README.md")],
        pending_interrupt=PendingInterruptContext(
            kind="question",
            request_id="question_1",
            selected_option="README.md",
            details={"source": "resume"},
        ),
        current_task="Summarize the architecture decisions.",
    )

    assert messages == (
        Message(role="system", content="You are agentlet."),
        Message(role="system", content="Durable memory:\n# Memory\n- Prefer tests."),
        Message(role="user", content="Read README.md"),
        Message(
            role="user",
            content=(
                "Interrupt resume context:\n"
                '{\n  "details": {\n    "source": "resume"\n  },\n'
                '  "kind": "question",\n'
                '  "request_id": "question_1",\n'
                '  "selected_option": "README.md"\n}'
            ),
        ),
        Message(role="user", content="Summarize the architecture decisions."),
    )


def test_context_builder_skips_blank_system_memory_and_task_content() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        system_instructions="   ",
        durable_memory="\n\n",
        current_task="  ",
        session_history=[Message(role="user", content="Only history remains.")],
    )

    assert messages == (Message(role="user", content="Only history remains."),)


def test_context_builder_preserves_significant_whitespace_in_text_inputs() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        system_instructions="  Keep leading spaces",
        durable_memory="    code block line\n  next line",
        current_task="  Finish the failing unit tests.",
    )

    assert messages == (
        Message(role="system", content="  Keep leading spaces"),
        Message(
            role="system",
            content="Durable memory:\n    code block line\n  next line",
        ),
        Message(role="user", content="  Finish the failing unit tests."),
    )


def test_context_builder_rejects_non_message_session_records() -> None:
    builder = ContextBuilder()

    with pytest.raises(ValueError, match="kind='message'"):
        builder.build(
            session_history=[
                SessionRecord(
                    record_id="meta_1",
                    kind="metadata",
                    payload={"turn_id": "turn_1"},
                )
            ]
        )


def test_current_task_state_rejects_empty_task() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        CurrentTaskState(task="")

    with pytest.raises(ValueError, match="must not be empty"):
        CurrentTaskState(task="   ")


def test_pending_interrupt_context_validates_question_payload() -> None:
    with pytest.raises(ValueError, match="require selected_option or free_text"):
        PendingInterruptContext(kind="question", request_id="question_1")

    with pytest.raises(ValueError, match="only valid for approval interrupts"):
        PendingInterruptContext(
            kind="question",
            request_id="question_1",
            selected_option="README.md",
            decision="approved",
        )


def test_pending_interrupt_context_validates_approval_payload() -> None:
    with pytest.raises(ValueError, match="require decision='approved' or 'rejected'"):
        PendingInterruptContext(kind="approval", request_id="approval_1")

    with pytest.raises(ValueError, match="only valid for question interrupts"):
        PendingInterruptContext(
            kind="approval",
            request_id="approval_1",
            decision="approved",
            free_text="extra",
        )


def test_context_builder_includes_approval_resume_context() -> None:
    builder = ContextBuilder()

    messages = builder.build(
        pending_interrupt=PendingInterruptContext(
            kind="approval",
            request_id="approval_1",
            decision="approved",
            comment="Looks safe.",
            details={"reviewed_by": "user"},
        )
    )

    assert messages == (
        Message(
            role="user",
            content=(
                "Interrupt resume context:\n"
                '{\n  "comment": "Looks safe.",\n'
                '  "decision": "approved",\n'
                '  "details": {\n    "reviewed_by": "user"\n  },\n'
                '  "kind": "approval",\n'
                '  "request_id": "approval_1"\n}'
            ),
        ),
    )

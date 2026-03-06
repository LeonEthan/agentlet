from __future__ import annotations

from dataclasses import asdict

import pytest

from agentlet.core.messages import Message, ToolCall
from agentlet.core.types import InterruptMetadata, InterruptOption, TokenUsage
from agentlet.llm.base import ModelClient
from agentlet.llm.schemas import (
    ModelRequest,
    ModelResponse,
    ModelToolDefinition,
    ToolChoice,
)
from agentlet.tools.base import Tool, ToolDefinition, ToolResult


def _read_definition() -> ToolDefinition:
    return ToolDefinition(
        name="Read",
        description="Read a file from the workspace.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        approval_category="read_only",
    )


def test_message_contract_serializes_tool_calls() -> None:
    message = Message(
        role="assistant",
        content="I need to inspect the file first.",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="Read",
                arguments={"path": "README.md"},
            )
        ],
        metadata={"turn_id": "turn_1"},
    )

    payload = asdict(message)

    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["arguments"] == {"path": "README.md"}
    assert payload["metadata"] == {"turn_id": "turn_1"}


def test_message_contract_round_trips_from_serialized_payload() -> None:
    original = Message(
        role="assistant",
        content="I need to inspect the file first.",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="Read",
                arguments={"path": "README.md"},
                metadata={"source": "model"},
            )
        ],
        metadata={"turn_id": "turn_1"},
    )

    restored = Message.from_dict(asdict(original))

    assert restored == original


def test_message_contract_rejects_invalid_tool_fields() -> None:
    with pytest.raises(ValueError, match="unsupported message role"):
        Message(role="invalid", content="nope")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="only assistant messages"):
        Message(
            role="user",
            content="run a tool",
            tool_calls=[ToolCall(id="call_1", name="Read")],
        )

    with pytest.raises(ValueError, match="tool messages must include tool_call_id"):
        Message(role="tool", content="tool output")


def test_tool_result_interrupt_metadata_is_normalized() -> None:
    interrupt = InterruptMetadata(
        kind="question",
        prompt="Which file should I edit?",
        request_id="interrupt_1",
        options=[
            InterruptOption(value="a", label="app.py"),
            InterruptOption(value="b", label="README.md"),
        ],
        allow_free_text=True,
        details={"source_tool": "AskUserQuestion"},
    )

    result = ToolResult.interrupt_result(
        output="Need clarification before proceeding.",
        interrupt=interrupt,
        metadata={"tool_name": "AskUserQuestion"},
    )

    assert result.interrupt is True
    assert result.is_error is False
    assert result.metadata == {
        "tool_name": "AskUserQuestion",
        "interrupt": {
            "kind": "question",
            "prompt": "Which file should I edit?",
            "request_id": "interrupt_1",
            "options": [
                {"value": "a", "label": "app.py"},
                {"value": "b", "label": "README.md"},
            ],
            "allow_free_text": True,
            "details": {"source_tool": "AskUserQuestion"},
        },
    }

    with pytest.raises(ValueError, match="must include metadata\\['interrupt'\\]"):
        ToolResult(output="Need clarification before proceeding.", interrupt=True)

    with pytest.raises(ValueError, match="requires interrupt=True"):
        ToolResult(
            output="Normal output",
            metadata={"interrupt": {"kind": "question", "prompt": "x"}},
        )


def test_model_request_validates_tool_choice_against_available_tools() -> None:
    definition = ModelToolDefinition.from_tool_definition(_read_definition())

    request = ModelRequest(
        messages=[Message(role="user", content="Read README.md")],
        tools=[definition],
        tool_choice=ToolChoice(mode="tool", tool_name="Read"),
    )

    assert request.tool_choice == ToolChoice(mode="tool", tool_name="Read")

    with pytest.raises(ValueError, match="must match an available tool"):
        ModelRequest(
            messages=[Message(role="user", content="Read README.md")],
            tools=[definition],
            tool_choice=ToolChoice(mode="tool", tool_name="Write"),
        )


def test_model_request_and_response_round_trip_without_approval_policy() -> None:
    request = ModelRequest(
        messages=[
            Message(
                role="assistant",
                content="I should read the file.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="Read",
                        arguments={"path": "README.md"},
                    )
                ],
            )
        ],
        tools=[ModelToolDefinition.from_tool_definition(_read_definition())],
        tool_choice=ToolChoice(mode="tool", tool_name="Read"),
        metadata={"request_id": "req_1"},
    )
    request_payload = asdict(request)

    assert "approval_category" not in request_payload["tools"][0]
    assert ModelRequest.from_dict(request_payload) == request

    response = ModelResponse(
        message=Message(
            role="assistant",
            content="Here is the result.",
            metadata={"provider": "fake"},
        ),
        finish_reason="stop",
        usage=TokenUsage(input_tokens=12, output_tokens=8),
        metadata={"model": "fake-model"},
    )

    assert ModelResponse.from_dict(asdict(response)) == response


def test_token_usage_normalizes_total_tokens() -> None:
    usage = TokenUsage(input_tokens=12, output_tokens=8)

    assert usage.total_tokens == 20

    with pytest.raises(ValueError, match="must equal input_tokens"):
        TokenUsage(input_tokens=1, output_tokens=2, total_tokens=99)


def test_protocols_are_runtime_checkable() -> None:
    class FakeTool:
        definition = _read_definition()

        def execute(self, arguments: dict[str, object]) -> ToolResult:
            return ToolResult(output=f"read {arguments['path']}")

    class FakeModelClient:
        def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                message=Message(role="assistant", content="done"),
                finish_reason="stop",
            )

    tool = FakeTool()
    client = FakeModelClient()

    assert isinstance(tool, Tool)
    assert isinstance(client, ModelClient)

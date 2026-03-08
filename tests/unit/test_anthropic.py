from __future__ import annotations

import pytest

from agentlet.core.messages import Message, ToolCall
from agentlet.llm.anthropic import (
    AnthropicModelClient,
    build_anthropic_request,
    parse_anthropic_response,
)
from agentlet.llm.schemas import ModelRequest, ModelToolDefinition, ToolChoice


def _read_tool() -> ModelToolDefinition:
    return ModelToolDefinition(
        name="Read",
        description="Read a file from the workspace.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


def test_build_anthropic_request_normalizes_system_tools_and_tool_results() -> None:
    request = ModelRequest(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="system", content="Prefer concise answers."),
            Message(role="user", content="Inspect README.md"),
            Message(
                role="assistant",
                content="I will inspect it.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="Read",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
            Message(role="tool", content="# README", tool_call_id="call_1"),
        ],
        tools=[_read_tool()],
        tool_choice=ToolChoice(mode="required"),
    )

    payload = build_anthropic_request(
        request,
        model="claude-test",
        max_output_tokens=512,
        request_defaults={"temperature": 0},
    )

    assert payload["model"] == "claude-test"
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0
    assert payload["system"] == "You are helpful.\n\nPrefer concise answers."
    assert payload["messages"] == [
        {"role": "user", "content": "Inspect README.md"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will inspect it."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "# README",
                }
            ],
        },
    ]
    assert payload["tools"] == [
        {
            "name": "Read",
            "description": "Read a file from the workspace.",
            "input_schema": _read_tool().input_schema,
        }
    ]
    assert payload["tool_choice"] == {"type": "any"}


def test_build_anthropic_request_merges_tool_results_with_following_user_text() -> None:
    request = ModelRequest(
        messages=[
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="Read",
                        arguments={"path": "README.md"},
                    )
                ],
            ),
            Message(role="tool", content="# README", tool_call_id="call_1"),
            Message(role="user", content="Continue the task."),
        ],
    )

    payload = build_anthropic_request(
        request,
        model="claude-test",
        max_output_tokens=512,
    )

    assert payload["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "Read",
                    "input": {"path": "README.md"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "# README",
                },
                {
                    "type": "text",
                    "text": "Continue the task.",
                },
            ],
        },
    ]


def test_parse_anthropic_response_normalizes_text_and_tool_use() -> None:
    response = parse_anthropic_response(
        {
            "id": "msg_1",
            "model": "claude-test",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "I need to inspect a file. "},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"path": "README.md"},
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 7,
            },
        }
    )

    assert response.message == Message(
        role="assistant",
        content="I need to inspect a file. ",
        tool_calls=[
            ToolCall(
                id="toolu_1",
                name="Read",
                arguments={"path": "README.md"},
                metadata={"provider_type": "tool_use"},
            )
        ],
    )
    assert response.finish_reason == "tool_calls"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 7
    assert response.usage.total_tokens == 17
    assert response.metadata == {
        "provider": "anthropic",
        "stop_reason": "tool_use",
        "response_id": "msg_1",
        "model": "claude-test",
    }


def test_parse_anthropic_response_rejects_unsupported_content_blocks() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported anthropic assistant content block type",
    ):
        parse_anthropic_response(
            {
                "stop_reason": "end_turn",
                "content": [{"type": "image", "source": {}}],
            }
        )


def test_parse_anthropic_response_treats_refusal_as_terminal() -> None:
    response = parse_anthropic_response(
        {
            "stop_reason": "refusal",
            "content": [{"type": "text", "text": "I can't help with that."}],
        }
    )

    assert response.finish_reason == "stop"
    assert response.message.content == "I can't help with that."


def test_anthropic_client_calls_transport_with_built_payload() -> None:
    captured_payload: dict[str, object] = {}

    def transport(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "Done",
                }
            ],
        }

    client = AnthropicModelClient(
        model="claude-test",
        max_output_tokens=256,
        transport=transport,
        request_defaults={"temperature": 0},
    )

    response = client.complete(
        ModelRequest(messages=[Message(role="user", content="Say done")])
    )

    assert captured_payload == {
        "temperature": 0,
        "model": "claude-test",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "Say done"}],
    }
    assert response.message.content == "Done"

from __future__ import annotations

import pytest

from agentlet.core.messages import Message, ToolCall
from agentlet.llm.openai_like import (
    OpenAILikeModelClient,
    build_openai_like_request,
    parse_openai_like_response,
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


def test_build_openai_like_request_normalizes_messages_and_tools() -> None:
    request = ModelRequest(
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Read README.md"),
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
            Message(
                role="tool",
                content="# README",
                tool_call_id="call_1",
            ),
        ],
        tools=[_read_tool()],
        tool_choice=ToolChoice(mode="tool", tool_name="Read"),
    )

    payload = build_openai_like_request(
        request,
        model="gpt-test",
        request_defaults={"temperature": 0},
    )

    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0
    assert payload["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert payload["messages"][2]["content"] is None
    assert payload["messages"][2]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "Read",
                "arguments": '{"path":"README.md"}',
            },
        }
    ]
    assert payload["messages"][3] == {
        "role": "tool",
        "content": "# README",
        "tool_call_id": "call_1",
    }
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file from the workspace.",
                "parameters": _read_tool().input_schema,
            },
        }
    ]
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": "Read"},
    }


def test_parse_openai_like_response_normalizes_final_message() -> None:
    response = parse_openai_like_response(
        {
            "id": "resp_1",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Done. "},
                            {"type": "output_text", "text": {"value": "README reviewed."}},
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }
    )

    assert response.message == Message(
        role="assistant",
        content="Done. README reviewed.",
        metadata={"choice_index": 0},
    )
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 8
    assert response.usage.total_tokens == 20
    assert response.metadata == {
        "response_id": "resp_1",
        "model": "gpt-test",
        "choice_index": 0,
    }


def test_parse_openai_like_response_normalizes_tool_calls() -> None:
    response = parse_openai_like_response(
        {
            "id": "resp_2",
            "model": "gpt-test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path":"README.md","options":{"lines":[1,2]}}',
                                },
                            }
                        ],
                    },
                }
            ],
        }
    )

    assert response.message == Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="Read",
                arguments={
                    "path": "README.md",
                    "options": {"lines": [1, 2]},
                },
                metadata={"provider_type": "function"},
            )
        ],
        metadata={"choice_index": 0},
    )
    assert response.finish_reason == "tool_calls"


def test_parse_openai_like_response_rejects_invalid_tool_arguments() -> None:
    with pytest.raises(ValueError, match="tool call arguments must be valid JSON"):
        parse_openai_like_response(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": "{bad json}",
                                    },
                                }
                            ],
                        },
                    }
                ]
            }
        )


def test_openai_like_client_calls_transport_with_built_payload() -> None:
    captured_payload: dict[str, object] = {}

    def transport(payload: dict[str, object]) -> dict[str, object]:
        captured_payload.update(payload)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "Done",
                    },
                }
            ]
        }

    client = OpenAILikeModelClient(
        model="gpt-test",
        transport=transport,
        request_defaults={"temperature": 0},
    )

    response = client.complete(
        ModelRequest(messages=[Message(role="user", content="Say done")])
    )

    assert captured_payload == {
        "temperature": 0,
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "Say done"}],
    }
    assert response.message.content == "Done"

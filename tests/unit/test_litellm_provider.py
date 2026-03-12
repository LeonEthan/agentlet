from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agentlet.agent.context import Message
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.providers.registry import ProviderConfig
from agentlet.agent.tools.registry import ToolSpec


def test_litellm_provider_builds_request_and_normalizes_dict_response() -> None:
    captured: dict[str, object] = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        }

    provider = LiteLLMProvider(
        ProviderConfig(
            model="gpt-4o-mini",
            api_key="test-key",
            api_base="http://localhost:4000/v1",
            temperature=0.2,
            max_tokens=512,
        ),
        completion_func=fake_completion,
    )

    response = asyncio.run(
        provider.complete(
            [Message(role="user", content="hello")],
            tools=[
                ToolSpec(
                    name="echo",
                    description="Echo input.",
                    parameters={"type": "object", "properties": {}},
                )
            ],
        )
    )

    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "test-key"
    assert captured["api_base"] == "http://localhost:4000/v1"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 512
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo input.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assert response.content == "hello"
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 3


def test_litellm_provider_normalizes_tool_calls_from_object_response() -> None:
    async def fake_completion(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="echo",
                                    arguments='{"text":"hello"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )

    provider = LiteLLMProvider(
        ProviderConfig(model="gpt-4o-mini"),
        completion_func=fake_completion,
    )

    response = asyncio.run(provider.complete([Message(role="user", content="hello")]))

    assert response.content is None
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "echo"
    assert response.tool_calls[0].arguments_json == '{"text":"hello"}'


def test_litellm_provider_serializes_dict_tool_arguments_as_json() -> None:
    async def fake_completion(**kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "echo",
                                    "arguments": {"text": "hello"},
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    provider = LiteLLMProvider(
        ProviderConfig(model="gpt-4o-mini"),
        completion_func=fake_completion,
    )

    response = asyncio.run(provider.complete([Message(role="user", content="hello")]))

    assert response.tool_calls[0].arguments_json == '{"text": "hello"}'


def test_litellm_provider_normalizes_streaming_deltas_and_tool_calls() -> None:
    captured: dict[str, object] = {}

    async def fake_stream_completion(**kwargs):
        captured.update(kwargs)

        async def _stream():
            yield {
                "choices": [
                    {
                        "delta": {"content": "hel"},
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": "lo",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "echo",
                                        "arguments": '{"te',
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": 'xt":"hello"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "total_tokens": 3,
                },
            }

        return _stream()

    provider = LiteLLMProvider(
        ProviderConfig(model="gpt-4o-mini"),
        stream_completion_func=fake_stream_completion,
    )

    events = asyncio.run(
        _collect_stream_events(provider.stream_complete([Message(role="user", content="hello")]))
    )

    assert captured["stream"] is True
    assert [event.kind for event in events] == [
        "content_delta",
        "content_delta",
        "response_complete",
    ]
    final_response = events[-1].response
    assert final_response is not None
    assert final_response.content == "hello"
    assert final_response.finish_reason == "tool_calls"
    assert final_response.tool_calls[0].arguments_json == '{"text":"hello"}'
    assert final_response.usage is not None
    assert final_response.usage.total_tokens == 3


async def _collect_stream_events(stream) -> list:
    return [event async for event in stream]

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

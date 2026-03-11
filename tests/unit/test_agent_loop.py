from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from agentlet.agent.agent_loop import AgentLoop, MaxIterationsExceeded
from agentlet.agent.context import ToolCall
from agentlet.agent.providers.registry import LLMResponse
from agentlet.agent.tools.registry import ToolRegistry, ToolSpec


class FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.seen_messages: list[list[str]] = []

    async def complete(self, messages, tools=None, model=None, temperature=None, max_tokens=None):
        self.seen_messages.append([message.role for message in messages])
        return self._responses.pop(0)


@dataclass(frozen=True)
class EchoTool:
    spec: ToolSpec = ToolSpec(
        name="echo",
        description="Return the same text.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    async def execute(self, arguments: dict[str, str]) -> str:
        return arguments["text"]


def test_agent_loop_returns_direct_response() -> None:
    provider = FakeProvider([LLMResponse(content="done", finish_reason="stop")])
    loop = AgentLoop(provider=provider)

    result = asyncio.run(loop.run_turn("hello"))

    assert result.output == "done"
    assert result.iterations == 1
    assert provider.seen_messages == [["system", "user"]]


def test_agent_loop_executes_tool_calls() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"hello from tool"}',
                    ),
                ),
                finish_reason="tool_calls",
            ),
            LLMResponse(content="tool complete", finish_reason="stop"),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry([EchoTool()]),
    )

    result = asyncio.run(loop.run_turn("say hello"))

    assert result.output == "tool complete"
    assert [message.role for message in result.context.history] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert provider.seen_messages == [
        ["system", "user"],
        ["system", "user", "assistant", "tool"],
    ]


def test_agent_loop_enforces_max_iterations() -> None:
    provider = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(id="call-1", name="echo", arguments_json='{"text":"1"}'),
                ),
            ),
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(id="call-2", name="echo", arguments_json='{"text":"2"}'),
                ),
            ),
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry([EchoTool()]),
        max_iterations=2,
    )

    with pytest.raises(MaxIterationsExceeded):
        asyncio.run(loop.run_turn("loop forever"))


def test_agent_loop_does_not_mutate_context_when_provider_fails() -> None:
    class BoomProvider:
        async def complete(
            self,
            messages,
            tools=None,
            model=None,
            temperature=None,
            max_tokens=None,
        ):
            raise RuntimeError("boom")

    from agentlet.agent.context import Context

    context = Context(system_prompt="system")
    loop = AgentLoop(provider=BoomProvider())

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(loop.run_turn("hello", context=context))

    assert context.history == []


def test_agent_loop_does_not_mutate_context_when_tool_execution_fails() -> None:
    class BoomTool:
        spec: ToolSpec = ToolSpec(
            name="echo",
            description="Raise an error.",
            parameters={"type": "object", "properties": {}},
        )

        async def execute(self, arguments: dict[str, str]) -> str:
            raise RuntimeError("tool boom")

    from agentlet.agent.context import Context

    provider = FakeProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"hello from tool"}',
                    ),
                ),
                finish_reason="tool_calls",
            ),
        ]
    )
    context = Context(system_prompt="system")
    loop = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry([BoomTool()]),
    )

    with pytest.raises(RuntimeError, match="tool boom"):
        asyncio.run(loop.run_turn("say hello", context=context))

    assert context.history == []

from __future__ import annotations

import asyncio

import pytest

from agentlet.agent.agent_loop import AgentLoop, MaxIterationsExceeded
from agentlet.agent.context import Context, ToolCall
from agentlet.agent.providers.registry import LLMResponse, ProviderStreamEvent
from agentlet.agent.tools.registry import ToolRegistry
from conftest import EchoTool, FakeProvider


def test_agent_loop_returns_direct_response() -> None:
    provider = FakeProvider([LLMResponse(content="done", finish_reason="stop")])
    loop = AgentLoop(provider=provider)

    result = asyncio.run(loop.run_turn("hello"))

    assert result.output == "done"
    assert result.iterations == 1
    assert provider.seen_messages == [["system", "user"]]


def test_agent_loop_uses_small_default_max_iterations() -> None:
    loop = AgentLoop(provider=FakeProvider([]))

    assert loop.max_iterations == 8


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

    context = Context(system_prompt="system")
    loop = AgentLoop(provider=BoomProvider())

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(loop.run_turn("hello", context=context))

    assert context.history == []


def test_agent_loop_does_not_mutate_context_when_tool_execution_fails() -> None:
    class BoomTool:
        spec = EchoTool.spec

        async def execute(self, arguments: dict[str, str]) -> str:
            raise RuntimeError("tool boom")

    context = Context(system_prompt="system")
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
    loop = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry([BoomTool()]),
    )

    with pytest.raises(RuntimeError, match="tool boom"):
        asyncio.run(loop.run_turn("say hello", context=context))

    assert context.history == []


def test_agent_loop_emits_stream_events_and_commits_final_state() -> None:
    provider = FakeProvider(
        [],
        stream_events=[
            ProviderStreamEvent(kind="content_delta", text="hel"),
            ProviderStreamEvent(kind="content_delta", text="lo"),
            ProviderStreamEvent(
                kind="response_complete",
                response=LLMResponse(content="hello", finish_reason="stop"),
            ),
        ],
    )
    loop = AgentLoop(provider=provider)
    from agentlet.agent.agent_loop import TurnEvent

    events: list[TurnEvent] = []

    result = asyncio.run(
        loop.run_turn(
            "hello",
            event_sink=events.append,
            stream=True,
        )
    )

    assert result.output == "hello"
    assert [message.role for message in result.context.history] == ["user", "assistant"]
    assert [event.kind for event in events] == [
        "turn_started",
        "assistant_delta",
        "assistant_delta",
        "assistant_completed",
        "turn_completed",
    ]


def test_agent_loop_ignores_stale_tool_calls_on_final_response() -> None:
    class RecordingEchoTool:
        spec = EchoTool.spec

        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        async def execute(self, arguments: dict[str, str]) -> str:
            self.calls.append(arguments)
            return arguments["text"]

    tool = RecordingEchoTool()
    provider = FakeProvider(
        [
            LLMResponse(
                content="final answer",
                tool_calls=(
                    ToolCall(
                        id="call-1",
                        name="echo",
                        arguments_json='{"text":"stale"}',
                    ),
                ),
                finish_reason="stop",
            )
        ]
    )
    loop = AgentLoop(
        provider=provider,
        tool_registry=ToolRegistry([tool]),
    )

    result = asyncio.run(loop.run_turn("hello"))

    assert result.output == "final answer"
    assert tool.calls == []
    assert [message.role for message in result.context.history] == ["user", "assistant"]
    assert result.context.history[-1].tool_calls == ()

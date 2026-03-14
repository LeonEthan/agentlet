from __future__ import annotations

"""Core orchestration loop for a single agent turn."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentlet.agent.context import Context, ToolCall, ToolResult
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import LLMProvider, LLMResponse
from agentlet.agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentTurnResult:
    """Result returned after one complete user turn."""

    output: str
    context: Context
    iterations: int
    finish_reason: str | None = None


@dataclass(frozen=True)
class TurnEvent:
    """Normalized runtime signal exposed to the CLI layer."""

    kind: Literal[
        "turn_started",
        "assistant_delta",
        "assistant_completed",
        "tool_requested",
        "tool_started",
        "tool_completed",
        "turn_completed",
        "turn_failed",
    ]
    user_input: str | None = None
    text: str | None = None
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    result: AgentTurnResult | None = None
    error: BaseException | None = None


class MaxIterationsExceeded(RuntimeError):
    """Raised when the loop keeps requesting tools beyond the safety budget."""

    pass


class AgentLoop:
    """Coordinate context building, model calls, and optional tool execution."""

    def __init__(
        self,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 80,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.provider = provider
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt or build_system_prompt()
        self.max_iterations = max_iterations

    async def run_turn(
        self,
        user_input: str,
        *,
        context: Context | None = None,
        event_sink: Callable[[TurnEvent], None] | None = None,
        stream: bool = False,
    ) -> AgentTurnResult:
        """Run one user turn until the model produces a final assistant answer.

        If an existing context is supplied, we work on a shallow copy first and
        only write the final history back once the turn completes successfully.
        That keeps caller-owned state unchanged when the provider or a tool fails.
        """
        if not user_input.strip():
            raise ValueError("user_input must not be empty.")

        self._emit(event_sink, TurnEvent(kind="turn_started", user_input=user_input))

        try:
            base_context = context or Context(system_prompt=self.system_prompt)
            # Copy the history so partial progress does not leak into the caller's
            # context until the turn is known to be successful.
            active_context = Context(
                system_prompt=base_context.system_prompt,
                history=base_context.history,
            )
            messages = active_context.build_messages(user_input)
            tool_schemas = self.tool_registry.get_tool_schemas() or None

            for iteration in range(1, self.max_iterations + 1):
                response = await self._run_provider_turn(
                    messages=messages,
                    tools=tool_schemas,
                    event_sink=event_sink,
                    stream=stream,
                )
                active_context.add_assistant_message(
                    response.content,
                    list(response.tool_calls),
                )
                self._emit(
                    event_sink,
                    TurnEvent(
                        kind="assistant_completed",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    ),
                )

                if not response.tool_calls:
                    # Commit the successful turn back into the caller-provided
                    # context only after all provider/tool work has completed.
                    if context is not None:
                        context.history[:] = active_context.history
                    result = AgentTurnResult(
                        output=response.content or "",
                        context=context or active_context,
                        iterations=iteration,
                        finish_reason=response.finish_reason,
                    )
                    self._emit(
                        event_sink,
                        TurnEvent(kind="turn_completed", result=result),
                    )
                    return result

                for call in response.tool_calls:
                    self._emit(
                        event_sink,
                        TurnEvent(kind="tool_requested", tool_call=call),
                    )
                    self._emit(
                        event_sink,
                        TurnEvent(kind="tool_started", tool_call=call),
                    )
                    # Tool results become regular messages so the next model call can
                    # reason over them using the same message abstraction.
                    result = await self.tool_registry.execute(call)
                    active_context.add_tool_result(result)
                    self._emit(
                        event_sink,
                        TurnEvent(kind="tool_completed", tool_result=result),
                    )

                messages = active_context.build_messages()

            raise MaxIterationsExceeded(
                f"Agent loop exceeded max_iterations={self.max_iterations}."
            )
        except BaseException as exc:
            self._emit(event_sink, TurnEvent(kind="turn_failed", error=exc))
            raise

    async def _run_provider_turn(
        self,
        *,
        messages,
        tools,
        event_sink: Callable[[TurnEvent], None] | None,
        stream: bool,
    ) -> LLMResponse:
        if not stream:
            return await self.provider.complete(messages, tools=tools)

        final_response: LLMResponse | None = None
        async for event in self.provider.stream_complete(messages, tools=tools):
            if event.kind == "content_delta" and event.text:
                self._emit(
                    event_sink,
                    TurnEvent(kind="assistant_delta", text=event.text),
                )
            elif event.kind == "response_complete":
                final_response = event.response

        if final_response is None:
            raise RuntimeError("Provider stream did not produce a final response.")
        return final_response

    def _emit(
        self,
        event_sink: Callable[[TurnEvent], None] | None,
        event: TurnEvent,
    ) -> None:
        if event_sink is not None:
            event_sink(event)

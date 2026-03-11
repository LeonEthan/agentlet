from __future__ import annotations

from dataclasses import dataclass

from agentlet.agent.context import Context
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import LLMProvider
from agentlet.agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class AgentTurnResult:
    output: str
    context: Context
    iterations: int
    finish_reason: str | None = None


class MaxIterationsExceeded(RuntimeError):
    pass


class AgentLoop:
    def __init__(
        self,
        provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 8,
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
    ) -> AgentTurnResult:
        if not user_input.strip():
            raise ValueError("user_input must not be empty.")

        base_context = context or Context(system_prompt=self.system_prompt)
        active_context = Context(
            system_prompt=base_context.system_prompt,
            history=base_context.history,
        )
        messages = active_context.build_messages(user_input)
        tool_schemas = self.tool_registry.get_tool_schemas() or None

        for iteration in range(1, self.max_iterations + 1):
            response = await self.provider.complete(messages, tools=tool_schemas)
            active_context.add_assistant_message(
                response.content,
                list(response.tool_calls),
            )

            if not response.tool_calls:
                if context is not None:
                    context.history[:] = active_context.history
                return AgentTurnResult(
                    output=response.content or "",
                    context=context or active_context,
                    iterations=iteration,
                    finish_reason=response.finish_reason,
                )

            for call in response.tool_calls:
                result = await self.tool_registry.execute(call)
                active_context.add_tool_result(result)

            messages = active_context.build_messages()

        raise MaxIterationsExceeded(
            f"Agent loop exceeded max_iterations={self.max_iterations}."
        )

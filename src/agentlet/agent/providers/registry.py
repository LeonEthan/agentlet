from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from agentlet.agent.context import Message, ToolCall
from agentlet.agent.tools.registry import ToolSpec


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


ProviderFactory = Callable[[ProviderConfig], LLMProvider]


class ProviderRegistryError(RuntimeError):
    pass


class ProviderRegistry:
    def __init__(self, factories: dict[str, ProviderFactory] | None = None) -> None:
        self._factories = dict(factories or {})

    def create(self, config: ProviderConfig) -> LLMProvider:
        provider_name = config.name.lower()

        if not self._factories:
            from agentlet.agent.providers.litellm_provider import LiteLLMProvider

            self._factories["openai"] = LiteLLMProvider

        factory = self._factories.get(provider_name)
        if factory is None:
            raise ProviderRegistryError(f"Unsupported provider: {config.name}")
        return factory(config)

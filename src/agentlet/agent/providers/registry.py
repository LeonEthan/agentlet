from __future__ import annotations

"""Provider-facing contracts and a minimal runtime registry."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from agentlet.agent.context import Message, ToolCall
from agentlet.agent.tools.registry import ToolSpec


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token accounting returned by a provider, when available."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    """Provider output normalized into agentlet's internal shape."""

    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None


@dataclass(frozen=True)
class ProviderStreamEvent:
    """Normalized streaming event emitted by provider adapters."""

    kind: Literal["content_delta", "response_complete"]
    text: str | None = None
    response: LLMResponse | None = None


# Default provider configuration values
DEFAULT_PROVIDER: str = "openai"
DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.0


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration shared by provider adapters."""

    name: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int | None = None


class LLMProvider(Protocol):
    """Narrow interface the orchestration loop depends on."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...

    def stream_complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[ProviderStreamEvent]: ...


ProviderFactory = Callable[[ProviderConfig], LLMProvider]


class ProviderRegistryError(RuntimeError):
    """Raised when a configured provider name cannot be resolved."""

    pass


class ProviderRegistry:
    """Resolve provider names into concrete adapter instances.

    The registry stays intentionally small in phase 1. A lazy default keeps the
    common OpenAI-compatible path working without importing LiteLLM unless it is
    actually needed.
    """

    def __init__(self, factories: dict[str, ProviderFactory] | None = None) -> None:
        self._factories = dict(factories or {})

    def create(self, config: ProviderConfig) -> LLMProvider:
        provider_name = config.name.lower()

        if not self._factories:
            # Delay the import so simple unit tests that fake the provider do not
            # require LiteLLM to be imported eagerly.
            from agentlet.agent.providers.litellm_provider import LiteLLMProvider

            self._factories["openai"] = LiteLLMProvider

        factory = self._factories.get(provider_name)
        if factory is None:
            raise ProviderRegistryError(f"Unsupported provider: {config.name}")
        return factory(config)

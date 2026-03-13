from __future__ import annotations

import pytest

from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry, ProviderRegistryError


@pytest.mark.parametrize(
    "provider_name",
    [
        "openai",
        "anthropic",
        "azure",
        "gemini",
        "cohere",
        "together_ai",
        "groq",
        "mistral",
        "fireworks",
        "anyscale",
    ],
)
def test_provider_registry_creates_litellm_provider_for_documented_backends(
    provider_name: str,
) -> None:
    registry = ProviderRegistry()

    provider = registry.create(ProviderConfig(name=provider_name, model="test-model"))

    assert isinstance(provider, LiteLLMProvider)
    assert provider.config.name == provider_name


def test_provider_registry_rejects_unknown_provider() -> None:
    registry = ProviderRegistry()

    with pytest.raises(ProviderRegistryError, match="Unsupported provider: unknown"):
        registry.create(ProviderConfig(name="unknown", model="test-model"))

from __future__ import annotations

import pytest

from agentlet.runtime.app import (
    AnthropicConfig,
    OpenAIConfig,
    OpenAILikeConfig,
    load_model_provider_config,
)


def test_load_model_provider_config_defaults_to_openai() -> None:
    config = load_model_provider_config(
        {
            "AGENTLET_MODEL": "gpt-4.1-mini",
            "AGENTLET_API_KEY": "sk-test",
        }
    )

    assert isinstance(config, OpenAIConfig)
    assert config.model == "gpt-4.1-mini"


def test_load_model_provider_config_defaults_to_openai_like_when_base_url_is_set() -> None:
    config = load_model_provider_config(
        {
            "AGENTLET_MODEL": "llama",
            "AGENTLET_API_KEY": "sk-test",
            "AGENTLET_BASE_URL": "https://example.test/v1",
        }
    )

    assert isinstance(config, OpenAILikeConfig)
    assert config.base_url == "https://example.test/v1"


def test_load_model_provider_config_supports_anthropic() -> None:
    config = load_model_provider_config(
        {
            "AGENTLET_PROVIDER": "anthropic",
            "AGENTLET_MODEL": "claude-sonnet-4-5",
            "AGENTLET_API_KEY": "sk-ant-test",
            "AGENTLET_MAX_OUTPUT_TOKENS": "1024",
        }
    )

    assert isinstance(config, AnthropicConfig)
    assert config.max_output_tokens == 1024


def test_load_model_provider_config_accepts_openai_like_alias() -> None:
    config = load_model_provider_config(
        {
            "AGENTLET_MODEL": "llama",
            "AGENTLET_API_KEY": "sk-test",
            "AGENTLET_BASE_URL": "https://example.test/v1",
        },
        provider="openai-like",
    )

    assert isinstance(config, OpenAILikeConfig)


def test_load_model_provider_config_requires_max_output_tokens_for_anthropic() -> None:
    with pytest.raises(ValueError, match="AGENTLET_MAX_OUTPUT_TOKENS must be set"):
        load_model_provider_config(
            {
                "AGENTLET_PROVIDER": "anthropic",
                "AGENTLET_MODEL": "claude-sonnet-4-5",
                "AGENTLET_API_KEY": "sk-ant-test",
            }
        )

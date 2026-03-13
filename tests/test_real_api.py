"""Real API integration tests using `~/.agentlet/settings.json`.

This module tests the actual API calls with real credentials from the user-level
settings file.
Run these tests with: python -m pytest tests/test_real_api.py -v

Note: This file relies on conftest.py for sys.path setup. Run with:
    uv run python -m pytest tests/test_real_api.py -v

Or after bootstrapping local settings:
    agentlet init --model gpt-5.4
    # then edit ~/.agentlet/settings.json with api_key/api_base as needed
    uv run python -m pytest tests/test_real_api.py -v
"""

from __future__ import annotations

import os
import asyncio
import importlib.util
import pytest

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.context import Context, Message, ToolCall
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.tools.registry import ToolRegistry, ToolSpec
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import DEFAULT_MODEL, DEFAULT_PROVIDER
from agentlet.settings import resolve_settings_defaults, load_settings

pytestmark = pytest.mark.real_api

if os.getenv("AGENTLET_RUN_REAL_API_TESTS") != "1":
    pytest.skip("Set AGENTLET_RUN_REAL_API_TESTS=1 to run live API tests.", allow_module_level=True)

if importlib.util.find_spec("litellm") is None:
    pytest.skip("litellm is not installed in the active environment.", allow_module_level=True)

# Default provider name used across tests - use shared constants


def _get_env_config(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> ProviderConfig:
    """Create a ProviderConfig from the effective local settings.

    This helper reduces duplication across tests.
    """
    resolved_settings = resolve_settings_defaults(load_settings())
    return ProviderConfig(
        name=resolved_settings.provider or DEFAULT_PROVIDER,
        model=model or resolved_settings.model or DEFAULT_MODEL,
        api_key=resolved_settings.api_key,
        api_base=resolved_settings.api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TestSettingsLoading:
    """Test that local settings are properly loaded."""

    def test_api_key_loaded(self):
        """Verify API key settings are loaded from settings.json."""
        api_key = _get_env_config().api_key
        assert api_key, "API key not found in ~/.agentlet/settings.json"
        assert len(api_key) > 20, f"API key seems too short: {api_key[:10]}..."
        print(f"\n  API Key: {api_key[:15]}... (length: {len(api_key)})")

    def test_base_url_loaded(self):
        """Verify base URL settings are loaded when the provider uses one."""
        base_url = _get_env_config().api_base
        if base_url is not None:
            assert base_url.startswith("http"), f"Invalid base URL: {base_url}"
            print(f"\n  Base URL: {base_url}")
        else:
            print("\n  Base URL: not configured for this provider")

    def test_model_config_loaded(self):
        """Verify model settings are loaded from settings.json or built-in defaults."""
        model = _get_env_config().model
        assert model, "Model not found in ~/.agentlet/settings.json or built-in defaults"
        print(f"\n  Model: {model}")


class TestProviderConfig:
    """Test LiteLLM provider configuration."""

    def test_provider_config_creation(self):
        """Test ProviderConfig dataclass with effective local settings."""
        config = _get_env_config(temperature=0.7, max_tokens=100)

        assert config.name
        assert config.api_key is not None
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        print(f"\n  Config created successfully:")
        print(f"    - Provider: {config.name}")
        print(f"    - Model: {config.model}")
        print(f"    - API Base: {config.api_base or 'not configured'}")

    def test_registry_creates_litellm_provider(self):
        """Test ProviderRegistry creates LiteLLM provider."""
        config = _get_env_config()

        registry = ProviderRegistry()
        provider = registry.create(config)

        assert isinstance(provider, LiteLLMProvider)
        assert provider.config == config
        print(f"\n  Provider created: {type(provider).__name__}")


class TestLiteLLMProvider:
    """Test actual API calls with LiteLLM provider."""

    @pytest.fixture
    def provider(self):
        """Create a LiteLLM provider with real config."""
        config = _get_env_config(temperature=0.0, max_tokens=50)
        return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_simple_completion(self, provider):
        """Test a simple completion call."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Say 'Hello from API test' and nothing else."),
        ]

        response = await provider.complete(messages)

        assert response.content is not None
        assert len(response.content) > 0
        assert response.finish_reason in ["stop", "length", None]
        print(f"\n  Response: {response.content[:100]}...")
        print(f"  Finish reason: {response.finish_reason}")
        if response.usage:
            print(f"  Tokens: {response.usage.total_tokens} "
                  f"(prompt: {response.usage.prompt_tokens}, "
                  f"completion: {response.usage.completion_tokens})")

    @pytest.mark.asyncio
    async def test_completion_with_tools(self, provider):
        """Test completion with tool definitions."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the weather?"),
        ]

        tools = [
            ToolSpec(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state"
                        }
                    },
                    "required": ["location"]
                }
            )
        ]

        response = await provider.complete(messages, tools=tools)

        # Response may or may not include tool calls depending on the model
        assert response.content is not None or len(response.tool_calls) > 0
        print(f"\n  Content: {response.content[:100] if response.content else 'None'}...")
        print(f"  Tool calls: {len(response.tool_calls)}")
        if response.tool_calls:
            for call in response.tool_calls:
                print(f"    - {call.name}({call.arguments_json})")

    @pytest.mark.asyncio
    async def test_completion_temperature_variations(self):
        """Test completion with different temperature settings concurrently."""
        temperatures = [0.0, 0.5, 1.0]

        async def _test_temp(temp: float) -> tuple[float, str]:
            """Make a completion call for a specific temperature."""
            config = _get_env_config(temperature=temp, max_tokens=20)
            provider = LiteLLMProvider(config)

            messages = [
                Message(role="system", content="You are creative."),
                Message(role="user", content="Say hi"),
            ]

            response = await provider.complete(messages)
            return (temp, response.content)

        # Run all temperature tests concurrently for efficiency
        results = await asyncio.gather(*[_test_temp(t) for t in temperatures])

        for temp, content in results:
            print(f"\n  Temperature {temp}: {content[:50]}...")
            assert content is not None and len(content) > 0


class TestAgentLoop:
    """Test the agent loop with real API calls."""

    @pytest.fixture
    def agent_loop(self):
        """Create an AgentLoop with real provider."""
        config = _get_env_config(temperature=0.0, max_tokens=100)
        registry = ProviderRegistry()
        provider = registry.create(config)

        return AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            system_prompt=build_system_prompt(),
            max_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_simple_turn(self, agent_loop):
        """Test a simple agent turn."""
        result = await agent_loop.run_turn("Say 'Test passed' and nothing else.")

        assert isinstance(result, AgentTurnResult)
        assert result.output is not None
        assert len(result.output) > 0
        assert result.iterations == 1  # Should complete in one iteration
        assert result.finish_reason in ["stop", "length", None]
        print(f"\n  Output: {result.output}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Finish reason: {result.finish_reason}")

    @pytest.mark.asyncio
    async def test_context_accumulation(self, agent_loop):
        """Test that context accumulates across turns."""
        context = Context(system_prompt=build_system_prompt())

        # First turn
        result1 = await agent_loop.run_turn(
            "Remember this number: 42. Just say 'OK'.",
            context=context
        )
        assert result1.output is not None

        # Second turn - should remember
        result2 = await agent_loop.run_turn(
            "What number did I tell you to remember?",
            context=context
        )

        assert "42" in result2.output or "forty-two" in result2.output.lower()
        assert len(context.history) > 0
        print(f"\n  First turn: {result1.output}")
        print(f"  Second turn: {result2.output}")
        print(f"  History messages: {len(context.history)}")

    @pytest.mark.asyncio
    async def test_max_iterations_protection(self):
        """Test that max_iterations is enforced."""
        config = _get_env_config(temperature=0.0)
        provider = LiteLLMProvider(config)

        loop = AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            max_iterations=1,
        )

        result = await loop.run_turn("Say hello")
        assert result.iterations == 1
        print(f"\n  Max iterations enforced: {result.iterations}")


class TestToolExecution:
    """Test tool execution with real API calls."""

    @pytest.fixture
    def echo_tool(self):
        """Import echo tool from conftest to avoid duplication."""
        # Import from conftest - pytest makes this available
        import sys
        from pathlib import Path
        conftest_path = Path(__file__).parent / "conftest.py"
        spec = importlib.util.spec_from_file_location("conftest", conftest_path)
        conftest = importlib.util.module_from_spec(spec)
        sys.modules["conftest_local"] = conftest
        spec.loader.exec_module(conftest)
        return conftest.EchoTool()

    @pytest.mark.asyncio
    async def test_tool_registry_execution(self, echo_tool):
        """Test tool registry can execute tools."""
        registry = ToolRegistry()
        registry.register(echo_tool)

        call = ToolCall(
            id="test-123",
            name="echo",
            arguments_json='{"message": "hello world"}'
        )

        result = await registry.execute(call)

        assert result.tool_call_id == "test-123"
        assert result.name == "echo"
        assert "ECHO: hello world" in result.content
        print(f"\n  Tool result: {result.content}")


class TestErrorHandling:
    """Test error handling in real API scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test behavior with invalid API key."""
        resolved_settings = resolve_settings_defaults(load_settings())
        config = ProviderConfig(
            name=DEFAULT_PROVIDER,
            model=resolved_settings.model or DEFAULT_MODEL,
            api_key="invalid-key-12345",
            api_base=resolved_settings.api_base,
        )
        provider = LiteLLMProvider(config)

        messages = [
            Message(role="user", content="Hello"),
        ]

        # Should raise an error
        with pytest.raises(Exception) as exc_info:
            await provider.complete(messages)

        print(f"\n  Expected error: {type(exc_info.value).__name__}")
        print(f"  Message: {str(exc_info.value)[:100]}...")


class TestSystemIntegration:
    """Full system integration tests."""

    @pytest.mark.asyncio
    async def test_end_to_end_single_turn(self):
        """Test a complete end-to-end single turn."""
        # Setup
        config = _get_env_config(temperature=0.0, max_tokens=50)
        registry = ProviderRegistry()
        provider = registry.create(config)
        tool_registry = ToolRegistry()

        loop = AgentLoop(
            provider=provider,
            tool_registry=tool_registry,
            system_prompt=build_system_prompt(),
            max_iterations=5,
        )

        # Execute
        result = await loop.run_turn("Respond with exactly: 'Integration test passed'")

        # Verify
        assert result.output is not None
        assert "Integration test passed" in result.output
        assert result.iterations >= 1
        assert result.context is not None

        print(f"\n  ✓ Output: {result.output}")
        print(f"  ✓ Iterations: {result.iterations}")
        print(f"  ✓ Context history: {len(result.context.history)} messages")

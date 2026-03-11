"""Real API integration tests using .env configuration.

This module tests the actual API calls with real credentials from .env file.
Run these tests with: python -m pytest tests/test_real_api.py -v

Note: This file relies on conftest.py for sys.path setup. Run with:
    uv run python -m pytest tests/test_real_api.py -v

Or with environment loaded:
    export $(cat .env | xargs) && uv run python -m pytest tests/test_real_api.py -v
"""

from __future__ import annotations

import os
import asyncio
import pytest

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.context import Context, Message, ToolCall
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.tools.registry import ToolRegistry, ToolSpec
from agentlet.agent.prompts.system_prompt import build_system_prompt

# Default provider name used across tests
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL_FALLBACK = "gpt-4o-mini"


def _get_env_config(
    temperature: float = 0.0,
    max_tokens: int | None = None,
    model: str | None = None,
) -> ProviderConfig:
    """Create a ProviderConfig from environment variables.

    This helper reduces duplication across tests.
    """
    return ProviderConfig(
        name=DEFAULT_PROVIDER,
        model=model or os.getenv("AGENTLET_MODEL", DEFAULT_MODEL_FALLBACK),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


class TestEnvLoading:
    """Test that environment variables are properly loaded."""

    def test_api_key_loaded(self):
        """Verify OPENAI_API_KEY is loaded from .env."""
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "OPENAI_API_KEY not found in environment"
        assert len(api_key) > 20, f"API key seems too short: {api_key[:10]}..."
        print(f"\n  API Key: {api_key[:15]}... (length: {len(api_key)})")

    def test_base_url_loaded(self):
        """Verify OPENAI_BASE_URL is loaded from .env."""
        base_url = os.getenv("OPENAI_BASE_URL")
        assert base_url, "OPENAI_BASE_URL not found in environment"
        assert base_url.startswith("http"), f"Invalid base URL: {base_url}"
        print(f"\n  Base URL: {base_url}")

    def test_model_config_loaded(self):
        """Verify AGENTLET_MODEL is loaded from .env."""
        model = os.getenv("AGENTLET_MODEL")
        assert model, "AGENTLET_MODEL not found in environment"
        print(f"\n  Model: {model}")


class TestProviderConfig:
    """Test LiteLLM provider configuration."""

    def test_provider_config_creation(self):
        """Test ProviderConfig dataclass with env vars."""
        config = _get_env_config(temperature=0.7, max_tokens=100)

        assert config.name == DEFAULT_PROVIDER
        assert config.api_key is not None
        assert config.api_base is not None
        assert config.temperature == 0.7
        assert config.max_tokens == 100
        print(f"\n  Config created successfully:")
        print(f"    - Provider: {config.name}")
        print(f"    - Model: {config.model}")
        print(f"    - API Base: {config.api_base}")

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
        """Create a simple echo tool for testing."""
        from dataclasses import dataclass

        @dataclass
        class EchoTool:
            spec = ToolSpec(
                name="echo",
                description="Echo the input message back",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo"
                        }
                    },
                    "required": ["message"]
                }
            )

            async def execute(self, arguments: dict) -> str:
                msg = arguments.get("message", "")
                return f"ECHO: {msg}"

        return EchoTool()

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
        config = ProviderConfig(
            name=DEFAULT_PROVIDER,
            model=os.getenv("AGENTLET_MODEL", DEFAULT_MODEL_FALLBACK),
            api_key="invalid-key-12345",
            api_base=os.getenv("OPENAI_BASE_URL"),
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

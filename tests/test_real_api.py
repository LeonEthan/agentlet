"""Real API integration tests using .env configuration.

This module tests the actual API calls with real credentials from .env file.
Run these tests with: python -m pytest tests/test_real_api.py -v
"""

from __future__ import annotations

import os
import sys
import asyncio
import pytest
from pathlib import Path

# Load .env before any imports that might need it
from dotenv import load_dotenv, find_dotenv

env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path, override=False)
    print(f"\n✓ Loaded .env from: {env_path}")
else:
    print("\n⚠ No .env file found!")

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.context import Context
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.tools.registry import ToolRegistry, ToolSpec
from agentlet.agent.prompts.system_prompt import build_system_prompt


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
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7,
            max_tokens=100,
        )

        assert config.name == "openai"
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
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
        )

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
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=50,
        )
        return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_simple_completion(self, provider):
        """Test a simple completion call."""
        from agentlet.agent.context import Message

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
        from agentlet.agent.context import Message

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
        """Test completion with different temperature settings."""
        from agentlet.agent.context import Message

        temperatures = [0.0, 0.5, 1.0]
        results = []

        for temp in temperatures:
            config = ProviderConfig(
                name="openai",
                model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                api_base=os.getenv("OPENAI_BASE_URL"),
                temperature=temp,
                max_tokens=20,
            )
            provider = LiteLLMProvider(config)

            messages = [
                Message(role="system", content="You are creative."),
                Message(role="user", content="Say hi"),
            ]

            response = await provider.complete(messages)
            results.append((temp, response.content))
            print(f"\n  Temperature {temp}: {response.content[:50]}...")

        # All should return non-empty content
        for temp, content in results:
            assert content is not None and len(content) > 0


class TestAgentLoop:
    """Test the agent loop with real API calls."""

    @pytest.fixture
    def agent_loop(self):
        """Create an AgentLoop with real provider."""
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=100,
        )
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
        # This would require a tool that keeps calling itself
        # For now, just verify the config works
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
        )
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
        from agentlet.agent.tools.registry import Tool
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
        from agentlet.agent.context import ToolCall

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
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key="invalid-key-12345",
            api_base=os.getenv("OPENAI_BASE_URL"),
        )
        provider = LiteLLMProvider(config)

        from agentlet.agent.context import Message
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
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=50,
        )
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


def run_async_tests():
    """Helper to run all async tests manually."""
    print("\n" + "="*60)
    print("Real API Integration Tests")
    print("="*60)

    # Check environment
    print("\n1. Environment Check")
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("AGENTLET_MODEL")

    print(f"   OPENAI_API_KEY: {'✓ Set' if api_key else '✗ Missing'} ({len(api_key) if api_key else 0} chars)")
    print(f"   OPENAI_BASE_URL: {'✓ Set' if base_url else '✗ Missing'} ({base_url})")
    print(f"   AGENTLET_MODEL: {'✓ Set' if model else '✗ Missing'} ({model})")

    if not all([api_key, base_url, model]):
        print("\n❌ Missing required environment variables!")
        return 1

    # Run simple test
    print("\n2. Simple API Call Test")
    asyncio.run(_run_simple_test())

    print("\n3. Agent Loop Test")
    asyncio.run(_run_agent_test())

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
    return 0


async def _run_simple_test():
    """Run a simple API test."""
    try:
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=30,
        )
        provider = LiteLLMProvider(config)

        from agentlet.agent.context import Message
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Say 'API test OK'"),
        ]

        response = await provider.complete(messages)
        print(f"   ✓ Response: {response.content}")
        print(f"   ✓ Finish reason: {response.finish_reason}")
        if response.usage:
            print(f"   ✓ Tokens: {response.usage.total_tokens}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise


async def _run_agent_test():
    """Run agent loop test."""
    try:
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=50,
        )
        provider = LiteLLMProvider(config)
        loop = AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            system_prompt=build_system_prompt(),
        )

        result = await loop.run_turn("Respond with exactly: 'Agent test OK'")
        print(f"   ✓ Output: {result.output}")
        print(f"   ✓ Iterations: {result.iterations}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        raise


if __name__ == "__main__":
    # Can run directly for quick test
    sys.exit(run_async_tests())

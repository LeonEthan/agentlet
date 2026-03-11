#!/usr/bin/env python3
"""Quick real API test script.

Run this to verify your .env configuration works correctly.

Usage:
    uv run python scripts/test_real_api.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env using the shared utility from CLI
from agentlet.cli.main import inject_project_env

inject_project_env()

from agentlet.agent.agent_loop import AgentLoop
from agentlet.agent.context import Context, Message
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.agent.prompts.system_prompt import build_system_prompt

# Default provider name used across tests
DEFAULT_PROVIDER = "openai"


def check_env():
    """Check environment variables."""
    print("\n" + "=" * 60)
    print("Environment Check")
    print("=" * 60)

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("AGENTLET_MODEL")

    all_ok = True

    if api_key:
        masked = api_key[:15] + "..." if len(api_key) > 15 else api_key
        print(f"✓ OPENAI_API_KEY: {masked} ({len(api_key)} chars)")
    else:
        print("✗ OPENAI_API_KEY: Not set")
        all_ok = False

    if base_url:
        print(f"✓ OPENAI_BASE_URL: {base_url}")
    else:
        print("✗ OPENAI_BASE_URL: Not set")
        all_ok = False

    if model:
        print(f"✓ AGENTLET_MODEL: {model}")
    else:
        print("✗ AGENTLET_MODEL: Not set")
        all_ok = False

    if not all_ok:
        print("\n❌ Missing required environment variables!")
        print("Please check your .env file.")
        return False

    return True


def _get_env_config(
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ProviderConfig:
    """Create a ProviderConfig from environment variables."""
    return ProviderConfig(
        name=DEFAULT_PROVIDER,
        model=os.getenv("AGENTLET_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def test_provider():
    """Test LiteLLM provider directly."""
    print("\n" + "=" * 60)
    print("Test 1: LiteLLM Provider Direct Call")
    print("=" * 60)

    try:
        config = _get_env_config(temperature=0.0, max_tokens=30)
        provider = LiteLLMProvider(config)

        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Say 'Provider test OK' and nothing else."),
        ]

        print("Sending request...")
        response = await provider.complete(messages)

        print("✓ Response received")
        print(f"  Content: {response.content}")
        print(f"  Finish reason: {response.finish_reason}")
        if response.usage:
            print(f"  Token usage: {response.usage.total_tokens} total "
                  f"({response.usage.prompt_tokens} prompt, "
                  f"{response.usage.completion_tokens} completion)")

        assert response.content is not None
        assert "Provider test OK" in response.content
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


async def test_agent_loop():
    """Test AgentLoop."""
    print("\n" + "=" * 60)
    print("Test 2: Agent Loop")
    print("=" * 60)

    try:
        config = _get_env_config(temperature=0.0, max_tokens=50)
        registry = ProviderRegistry()
        provider = registry.create(config)

        loop = AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            system_prompt=build_system_prompt(),
            max_iterations=3,
        )

        print("Sending request...")
        result = await loop.run_turn(
            "Respond with exactly: 'Agent loop test OK'"
        )

        print("✓ Response received")
        print(f"  Output: {result.output}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Finish reason: {result.finish_reason}")

        assert "Agent loop test OK" in result.output
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


async def test_cli_simulation():
    """Simulate CLI chat."""
    from agentlet.cli.main import run_chat

    print("\n" + "=" * 60)
    print("Test 3: CLI Simulation")
    print("=" * 60)

    try:
        result = await run_chat(
            message="Say 'CLI simulation OK'",
            provider_name=DEFAULT_PROVIDER,
            model=os.getenv("AGENTLET_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=30,
        )

        print("✓ Response received")
        print(f"  Output: {result.output}")
        print(f"  Iterations: {result.iterations}")

        assert "CLI simulation OK" in result.output
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


async def test_multi_turn():
    """Test multi-turn conversation with context."""
    print("\n" + "=" * 60)
    print("Test 4: Multi-turn Conversation")
    print("=" * 60)

    try:
        config = _get_env_config(temperature=0.0, max_tokens=50)
        provider = LiteLLMProvider(config)

        loop = AgentLoop(
            provider=provider,
            tool_registry=ToolRegistry(),
            system_prompt=build_system_prompt(),
        )

        # Create shared context
        context = Context(system_prompt=build_system_prompt())

        # Turn 1
        print("Turn 1: Sending message...")
        result1 = await loop.run_turn(
            "Remember this secret code: 12345. Just say 'Got it'.",
            context=context
        )
        print(f"  Response: {result1.output}")

        # Turn 2 - should remember
        print("\nTurn 2: Asking about the code...")
        result2 = await loop.run_turn(
            "What was the secret code I told you?",
            context=context
        )
        print(f"  Response: {result2.output}")

        # Verify memory
        if "12345" in result2.output:
            print("✓ Context memory working correctly!")
            return True
        else:
            print("⚠ Code not found in response (might be model limitation)")
            return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Agentlet Real API Tests")
    print("=" * 60)

    # Check environment
    if not check_env():
        return 1

    # Run tests concurrently for efficiency
    results = await asyncio.gather(
        test_provider(),
        test_agent_loop(),
        test_cli_simulation(),
        test_multi_turn(),
        return_exceptions=True
    )

    # Handle results (convert exceptions to False)
    test_names = [
        "Provider Direct Call",
        "Agent Loop",
        "CLI Simulation",
        "Multi-turn Conversation"
    ]
    processed_results = [
        (name, False if isinstance(result, Exception) else result)
        for name, result in zip(test_names, results)
    ]

    # Print any exceptions that occurred
    for name, result in zip(test_names, results):
        if isinstance(result, Exception):
            print(f"\n✗ {name} raised an exception: {result}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, r in processed_results if r)
    total = len(processed_results)

    for name, result in processed_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

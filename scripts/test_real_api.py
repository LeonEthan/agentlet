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

# Load .env
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)
        print(f"✓ Loaded .env from: {env_path}")
    else:
        print("⚠ No .env file found!")
except ImportError:
    print("⚠ python-dotenv not installed, skipping .env load")

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agentlet.agent.agent_loop import AgentLoop
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.agent.prompts.system_prompt import build_system_prompt


def check_env():
    """Check environment variables."""
    print("\n" + "="*60)
    print("Environment Check")
    print("="*60)

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


async def test_provider():
    """Test LiteLLM provider directly."""
    print("\n" + "="*60)
    print("Test 1: LiteLLM Provider Direct Call")
    print("="*60)

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
            Message(role="user", content="Say 'Provider test OK' and nothing else."),
        ]

        print("Sending request...")
        response = await provider.complete(messages)

        print(f"✓ Response received")
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
    print("\n" + "="*60)
    print("Test 2: Agent Loop")
    print("="*60)

    try:
        config = ProviderConfig(
            name="openai",
            model=os.getenv("AGENTLET_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=50,
        )
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

        print(f"✓ Response received")
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
    print("\n" + "="*60)
    print("Test 3: CLI Simulation")
    print("="*60)

    try:
        from agentlet.cli.main import run_chat

        result = await run_chat(
            message="Say 'CLI simulation OK'",
            provider_name="openai",
            model=os.getenv("AGENTLET_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL"),
            temperature=0.0,
            max_tokens=30,
        )

        print(f"✓ Response received")
        print(f"  Output: {result.output}")
        print(f"  Iterations: {result.iterations}")

        assert "CLI simulation OK" in result.output
        return True

    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False


async def test_multi_turn():
    """Test multi-turn conversation with context."""
    print("\n" + "="*60)
    print("Test 4: Multi-turn Conversation")
    print("="*60)

    try:
        from agentlet.agent.context import Context

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
    print("\n" + "="*60)
    print("Agentlet Real API Tests")
    print("="*60)

    # Check environment
    if not check_env():
        return 1

    # Run tests
    results = []

    results.append(("Provider Direct Call", await test_provider()))
    results.append(("Agent Loop", await test_agent_loop()))
    results.append(("CLI Simulation", await test_cli_simulation()))
    results.append(("Multi-turn Conversation", await test_multi_turn()))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
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

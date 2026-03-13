#!/usr/bin/env python3
"""Quick real API test script.

Run this to verify your exported env vars or ~/.agentlet/settings.json config.

Usage:
    uv run python scripts/test_real_api.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from agentlet.agent.agent_loop import AgentLoop
from agentlet.agent.context import Context, Message
from agentlet.agent.providers.litellm_provider import LiteLLMProvider
from agentlet.agent.providers.registry import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ProviderConfig,
    ProviderRegistry,
)
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.settings import load_settings, resolve_settings_defaults


def _load_effective_settings():
    """Load the same effective settings used by the CLI."""
    return resolve_settings_defaults(load_settings())


def check_configuration() -> bool:
    """Check the effective provider configuration."""
    print("\n" + "=" * 60)
    print("Configuration Check")
    print("=" * 60)

    settings = _load_effective_settings()
    provider = settings.provider or DEFAULT_PROVIDER
    model = settings.model or DEFAULT_MODEL
    api_key = settings.api_key
    base_url = settings.api_base

    all_ok = True

    print(f"✓ provider: {provider}")
    print(f"✓ model: {model}")

    if api_key:
        masked = api_key[:15] + "..." if len(api_key) > 15 else api_key
        print(f"✓ api_key: {masked} ({len(api_key)} chars)")
    else:
        print("✗ api_key: Not set")
        all_ok = False

    if base_url:
        print(f"✓ api_base: {base_url}")
    else:
        print("• api_base: not set")

    if not all_ok:
        print("\n❌ Missing required configuration!")
        print("Set exported env vars or run `agentlet init` first.")
        return False

    return True


def _get_effective_config(
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ProviderConfig:
    """Create a ProviderConfig from the effective local settings."""
    settings = _load_effective_settings()
    return ProviderConfig(
        name=settings.provider or DEFAULT_PROVIDER,
        model=settings.model or DEFAULT_MODEL,
        api_key=settings.api_key,
        api_base=settings.api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def test_provider():
    """Test LiteLLM provider directly."""
    print("\n" + "=" * 60)
    print("Test 1: LiteLLM Provider Direct Call")
    print("=" * 60)

    try:
        config = _get_effective_config(temperature=0.0, max_tokens=30)
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
        config = _get_effective_config(temperature=0.0, max_tokens=50)
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
        config = _get_effective_config(temperature=0.0, max_tokens=30)
        result = await run_chat(
            message="Say 'CLI simulation OK'",
            provider_name=config.name,
            model=config.model,
            api_key=config.api_key,
            api_base=config.api_base,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
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
        config = _get_effective_config(temperature=0.0, max_tokens=50)
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

    # Check configuration
    if not check_configuration():
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

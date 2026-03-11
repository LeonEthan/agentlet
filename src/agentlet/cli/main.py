from __future__ import annotations

"""CLI entrypoint for local agentlet experiments."""

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv, find_dotenv

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.tools.registry import ToolRegistry


def inject_project_env() -> None:
    """Load .env file from current or parent directories into os.environ.

    Uses load_dotenv() with override=False to avoid overwriting existing
    environment variables.
    """
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=False)


def build_parser() -> argparse.ArgumentParser:
    """Build the small phase-1 CLI surface."""
    parser = argparse.ArgumentParser(prog="agentlet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="Run a single-turn agent chat.")
    chat.add_argument("message", nargs="?", help="User message. Reads from stdin when omitted.")
    chat.add_argument("--provider", default="openai", help="Provider name.")
    chat.add_argument(
        "--model",
        default=os.getenv("AGENTLET_MODEL", "gpt-4o-mini"),
        help="Model name.",
    )
    chat.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY"),
        help="Provider API key.",
    )
    chat.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional OpenAI-compatible base URL.",
    )
    chat.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override.",
    )
    return parser


async def run_chat(
    *,
    message: str,
    provider_name: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    temperature: float,
    max_tokens: int | None,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
) -> AgentTurnResult:
    """Wire CLI arguments into runtime dependencies and execute one turn."""
    registry = provider_registry or ProviderRegistry()
    provider = registry.create(
        ProviderConfig(
            name=provider_name,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    loop = AgentLoop(
        provider=provider,
        tool_registry=tool_registry or ToolRegistry(),
        system_prompt=build_system_prompt(),
    )
    return await loop.run_turn(message)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI input, run the requested command, and print the final output."""
    inject_project_env()
    # Load .env before building parser defaults so environment-backed defaults
    # are visible to argparse immediately.
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "chat":
        parser.error(f"Unsupported command: {args.command}")

    raw_message = args.message if args.message is not None else sys.stdin.read()
    message = raw_message.strip()
    if not message:
        parser.error("A message is required via argv or stdin.")

    # asyncio.run keeps the CLI entrypoint synchronous while the runtime stays
    # fully async-friendly internally.
    result = asyncio.run(
        run_chat(
            message=message,
            provider_name=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    print(result.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

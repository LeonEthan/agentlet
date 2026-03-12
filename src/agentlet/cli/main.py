from __future__ import annotations

"""CLI entrypoint for local agentlet experiments."""

import argparse
import os
import sys

from dotenv import load_dotenv, find_dotenv

from agentlet.agent.providers.registry import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    ProviderRegistry,
)
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.cli.chat_app import ChatCLIError, run_chat_command


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

    chat = subparsers.add_parser("chat", help="Run agentlet chat in one-shot or interactive mode.")
    chat.add_argument("message", nargs="?", help="User message. Reads from stdin when omitted.")
    mode_group = chat.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="Resume the latest interactive session in the current working directory.",
    )
    mode_group.add_argument(
        "--session",
        dest="session_id",
        help="Resume a specific interactive session by id.",
    )
    mode_group.add_argument(
        "--new-session",
        dest="new_session",
        action="store_true",
        help="Force a fresh interactive session.",
    )
    chat.add_argument(
        "--print",
        dest="print_mode",
        action="store_true",
        help="Force one-shot print mode even when stdin is a TTY.",
    )
    chat.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider name.")
    chat.add_argument(
        "--model",
        default=os.getenv("AGENTLET_MODEL", DEFAULT_MODEL),
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
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max_tokens override.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI input and dispatch to the requested command mode."""
    inject_project_env()
    # Load .env before building parser defaults so environment-backed defaults
    # are visible to argparse immediately.
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "chat":
        parser.error(f"Unsupported command: {args.command}")

    try:
        return run_chat_command(
            args,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            provider_registry=ProviderRegistry(),
            tool_registry=ToolRegistry(),
        )
    except ChatCLIError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

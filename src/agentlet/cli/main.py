from __future__ import annotations

"""CLI entrypoint for local agentlet experiments."""

import argparse
import sys
from pathlib import Path

from agentlet.agent.providers.registry import (
    ProviderRegistry,
)
from agentlet.settings import (
    AgentletSettings,
    SettingsError,
    canonical_settings_path,
    default_settings_path,
    load_settings,
    resolve_settings_defaults,
    write_settings,
)
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.cli.chat_app import ChatCLIError, run_chat_command


def build_parser(defaults: AgentletSettings) -> argparse.ArgumentParser:
    """Build the small phase-1 CLI surface."""
    parser = argparse.ArgumentParser(prog="agentlet")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser(
        "init",
        help="Create ~/.agentlet/settings.json with the current effective defaults.",
    )
    init.add_argument("--provider", default=defaults.provider, help="Provider name to store.")
    init.add_argument("--model", default=defaults.model, help="Model name to store.")
    init.add_argument("--api-key", default=defaults.api_key, help="Provider API key to store.")
    init.add_argument(
        "--api-base",
        default=defaults.api_base,
        help="Optional OpenAI-compatible base URL to store.",
    )
    init.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help="Default sampling temperature to store.",
    )
    init.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help="Default max_tokens override to store.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing settings file.",
    )

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
    chat.add_argument("--provider", default=defaults.provider, help="Provider name.")
    chat.add_argument(
        "--model",
        default=defaults.model,
        help="Model name.",
    )
    chat.add_argument(
        "--api-key",
        default=defaults.api_key,
        help="Provider API key.",
    )
    chat.add_argument(
        "--api-base",
        default=defaults.api_base,
        help="Optional OpenAI-compatible base URL.",
    )
    chat.add_argument(
        "--temperature",
        type=float,
        default=defaults.temperature,
        help="Sampling temperature.",
    )
    chat.add_argument(
        "--max-tokens",
        type=int,
        default=defaults.max_tokens,
        help="Optional max_tokens override.",
    )
    return parser


def main(argv: list[str] | None = None, *, home_dir: Path | None = None) -> int:
    """Parse CLI input and dispatch to the requested command mode."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    settings_path = default_settings_path(home_dir)
    write_path = canonical_settings_path(home_dir)
    settings_error: SettingsError | None = None
    try:
        stored_settings = load_settings(settings_path)
    except SettingsError as exc:
        stored_settings = AgentletSettings()
        settings_error = exc

    parser = build_parser(resolve_settings_defaults(stored_settings))
    args = parser.parse_args(raw_argv)

    if args.command == "init":
        try:
            written_path = write_settings(
                AgentletSettings(
                    provider=args.provider,
                    model=args.model,
                    api_key=args.api_key,
                    api_base=args.api_base,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                ),
                settings_path=write_path,
                force=args.force,
            )
        except SettingsError as exc:
            parser.error(str(exc))
        print(f"Wrote settings to {written_path}", file=sys.stdout)
        return 0

    if settings_error is not None:
        parser.error(str(settings_error))

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

from __future__ import annotations

"""CLI entrypoint for local agentlet experiments."""

import argparse
import sys
from pathlib import Path

from agentlet.agent.providers.registry import (
    ProviderRegistry,
)
from agentlet.agent.tools.policy import (
    DEFAULT_MAX_HTML_EXTRACT_BYTES,
    MAX_HTML_EXTRACT_BYTES_LIMIT,
    ToolPolicy,
    ToolRuntimeConfig,
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
from agentlet.agent.tools.builtins import build_default_registry
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.cli.chat_app import ChatCLIError, _settings_from_args, run_chat_command


def _int_in_range(minimum: int, maximum: int | None = None):
    """Build an argparse type that rejects integers outside the range."""

    def parse(raw_value: str) -> int:
        value = int(raw_value)
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be an integer >= {minimum}")
        if maximum is not None and value > maximum:
            raise argparse.ArgumentTypeError(
                f"must be an integer between {minimum} and {maximum}"
            )
        return value

    return parse


# Backward compatibility: _int_at_least is now a thin wrapper around _int_in_range
_int_at_least = _int_in_range


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
        "--max-iterations",
        type=_int_at_least(1),
        default=defaults.max_iterations,
        help="Default max provider/tool iterations per turn to store.",
    )
    init.add_argument(
        "--max-html-extract-bytes",
        type=_int_in_range(1, MAX_HTML_EXTRACT_BYTES_LIMIT),
        default=defaults.max_html_extract_bytes,
        help="Default HTML fetch byte budget to use during readable-text extraction.",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing settings file.",
    )

    chat = subparsers.add_parser("chat", help="Run agentlet chat in one-shot or interactive mode.")
    chat.add_argument("message", nargs="?", help="User message. Reads from stdin when omitted.")
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
    chat.add_argument(
        "--max-iterations",
        type=_int_at_least(1),
        default=defaults.max_iterations,
        help="Maximum provider/tool iterations per turn.",
    )
    chat.add_argument(
        "--max-html-extract-bytes",
        type=_int_in_range(1, MAX_HTML_EXTRACT_BYTES_LIMIT),
        default=defaults.max_html_extract_bytes,
        help="Byte budget used for HTML readable-text extraction before truncating output.",
    )
    chat.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve write, bash, and network tool actions for this run.",
    )
    chat.add_argument(
        "--deny-write",
        action="store_true",
        help="Disable Write and Edit tools.",
    )
    chat.add_argument(
        "--deny-bash",
        action="store_true",
        help="Disable Bash tool.",
    )
    chat.add_argument(
        "--deny-network",
        action="store_true",
        help="Disable WebSearch and WebFetch tools.",
    )
    return parser


def _provider_change_preserves_sensitive_settings(
    *,
    stored_settings: AgentletSettings,
    next_provider: str | None,
) -> bool:
    if not (stored_settings.api_key or stored_settings.api_base):
        return False
    previous_provider = resolve_settings_defaults(stored_settings).provider
    return previous_provider != next_provider


def _resolve_tool_policy(stored: AgentletSettings, args: argparse.Namespace) -> ToolPolicy:
    """Resolve tool policy from stored settings and CLI flags.

    Settings file values take precedence over CLI defaults.
    CLI flags override both.
    """

    def _resolve_field(setting_value: bool | None, deny_flag: bool) -> bool:
        if deny_flag:
            return False
        if setting_value is not None:
            return setting_value
        return True

    return ToolPolicy(
        allow_network=_resolve_field(stored.allow_network, args.deny_network),
        allow_write=_resolve_field(stored.allow_write, args.deny_write),
        allow_bash=_resolve_field(stored.allow_bash, args.deny_bash),
    )


def _build_tool_runtime(
    args: argparse.Namespace,
    *,
    settings: AgentletSettings,
    cwd: Path,
) -> ToolRuntimeConfig:
    """Build runtime config from effective chat settings and cwd."""
    chat_settings = _settings_from_args(args, fallback=settings)
    return ToolRuntimeConfig(
        cwd=cwd,
        max_html_extract_bytes=(
            chat_settings.max_html_extract_bytes
            if chat_settings.max_html_extract_bytes is not None
            else DEFAULT_MAX_HTML_EXTRACT_BYTES
        ),
    )


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

    effective_settings = resolve_settings_defaults(stored_settings)
    parser = build_parser(effective_settings)
    args = parser.parse_args(raw_argv)

    if args.command == "init":
        if _provider_change_preserves_sensitive_settings(
            stored_settings=stored_settings,
            next_provider=args.provider,
        ):
            parser.error(
                "Changing provider with stored api_key/api_base is not supported via CLI. "
                "Edit ~/.agentlet/settings.json manually to update or clear sensitive fields."
            )
        try:
            written_path = write_settings(
                AgentletSettings(
                    provider=args.provider,
                    model=args.model,
                    api_key=stored_settings.api_key,
                    api_base=stored_settings.api_base,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_iterations=args.max_iterations,
                    max_html_extract_bytes=args.max_html_extract_bytes,
                    allow_write=stored_settings.allow_write,
                    allow_bash=stored_settings.allow_bash,
                    allow_network=stored_settings.allow_network,
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

    # Build tool policy from settings + CLI args
    tool_policy = _resolve_tool_policy(stored_settings, args)

    # Build runtime config from cwd
    tool_runtime = _build_tool_runtime(args, settings=effective_settings, cwd=Path.cwd())

    # Build the default registry with enabled tools based on policy
    tool_registry = build_default_registry(tool_policy, tool_runtime)

    try:
        return run_chat_command(
            args,
            settings=effective_settings,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
            provider_registry=ProviderRegistry(),
            tool_registry=tool_registry,
        )
    except ChatCLIError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

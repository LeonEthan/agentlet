from __future__ import annotations

"""Mode selection and top-level chat command wiring."""

import asyncio
from pathlib import Path
from typing import TextIO

from rich.console import Console

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.context import Context
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    ProviderConfig,
    ProviderRegistry,
)
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.cli.presenter import ChatPresenter
from agentlet.cli.prompt import create_prompt_session
from agentlet.cli.repl import run_repl
from agentlet.cli.sessions import LoadedSession, SessionStore
from agentlet.settings import AgentletSettings


class ChatCLIError(ValueError):
    """Raised when CLI arguments resolve to an invalid chat mode."""


async def run_chat(
    message: str,
    settings: AgentletSettings,
    *,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
) -> AgentTurnResult:
    """Wire one-shot CLI arguments into runtime dependencies and execute one turn."""
    loop = _create_agent_loop(
        settings=settings,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        system_prompt=system_prompt,
    )
    return await loop.run_turn(message)


def _create_agent_loop(
    settings: AgentletSettings,
    *,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
    _config: ProviderConfig | None = None,
) -> AgentLoop:
    """Build one configured runtime loop for the current CLI invocation."""
    registry = provider_registry or ProviderRegistry()
    config = _config if _config is not None else _create_provider_config(settings)
    provider = registry.create(config)
    return AgentLoop(
        provider=provider,
        tool_registry=tool_registry or ToolRegistry(),
        system_prompt=system_prompt or build_system_prompt(),
    )


def _create_provider_config(settings: AgentletSettings) -> ProviderConfig:
    """Create a ProviderConfig from settings with defaults applied."""
    return ProviderConfig(
        name=settings.provider or DEFAULT_PROVIDER,
        model=settings.model or DEFAULT_MODEL,
        api_key=settings.api_key,
        api_base=settings.api_base,
        temperature=settings.temperature
        if settings.temperature is not None
        else DEFAULT_TEMPERATURE,
        max_tokens=settings.max_tokens,
    )


def run_chat_command(
    args,
    settings: AgentletSettings,
    *,
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    prompt_session=None,
    console: Console | None = None,
    cwd: Path | None = None,
    stdin_isatty: bool | None = None,
) -> int:
    """Execute the requested chat mode and return the process exit code."""
    message, interactive = _resolve_chat_mode(args, stdin=stdin, stdin_isatty=stdin_isatty)
    working_dir = cwd or Path.cwd()

    if not interactive:
        result = asyncio.run(
            run_chat(
                message=message or "",
                settings=settings,
                provider_registry=provider_registry,
                tool_registry=tool_registry,
            )
        )
        print(result.output, file=stdout)
        return 0

    # Interactive mode - always start fresh session (no --session support)
    session_store = SessionStore(working_dir)
    system_prompt = build_system_prompt()
    config = _create_provider_config(settings)
    loop = _create_agent_loop(
        settings=settings,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        system_prompt=system_prompt,
        _config=config,
    )
    prompt = prompt_session or create_prompt_session(session_store.history_path)
    info = session_store.start_session(
        provider_name=config.name,
        model=config.model,
        api_base=config.api_base,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        system_prompt=loop.system_prompt,
    )
    loaded_session = LoadedSession(
        info=info,
        context=Context(system_prompt=loop.system_prompt),
    )

    presenter = ChatPresenter(console or Console(file=stdout, stderr=False))
    return run_repl(
        loop=loop,
        prompt_input=prompt,
        presenter=presenter,
        session_store=session_store,
        cwd=working_dir,
        loaded_session=loaded_session,
    )


def _read_and_validate_message(source: TextIO) -> str:
    """Read from source, strip whitespace, and validate non-empty."""
    message = source.read().strip()
    if not message:
        raise ChatCLIError("A message is required via argv or stdin.")
    return message


def _resolve_chat_mode(
    args, *, stdin: TextIO, stdin_isatty: bool | None = None
) -> tuple[str | None, bool]:
    """Resolve CLI arguments and stdin shape into (message, interactive) tuple."""
    is_tty = stdin.isatty() if stdin_isatty is None else stdin_isatty

    if args.message is not None:
        message = args.message.strip()
        if not message:
            raise ChatCLIError("A message is required via argv or stdin.")
        return message, False

    if args.print_mode:
        if is_tty:
            raise ChatCLIError(
                "--print requires a message argument or redirected stdin."
            )
        return _read_and_validate_message(stdin), False

    if not is_tty:
        return _read_and_validate_message(stdin), False

    return None, True

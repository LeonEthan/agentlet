from __future__ import annotations

"""Mode selection and top-level chat command wiring."""

import asyncio
from pathlib import Path
from typing import Any
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
from agentlet.cli.sessions import LoadedSession, SessionError, SessionStore, load_session_for_resume
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
    chat_settings = _settings_from_args(args, fallback=settings)

    if not interactive:
        result = asyncio.run(
            run_chat(
                message=message or "",
                settings=chat_settings,
                provider_registry=provider_registry,
                tool_registry=tool_registry,
            )
        )
        print(result.output, file=stdout)
        return 0

    session_store = SessionStore(working_dir)
    try:
        loaded_session = load_session_for_resume(
            session_store,
            continue_session=args.continue_session,
            session_id=args.session_id,
        )
    except SessionError as exc:
        raise ChatCLIError(str(exc)) from exc

    resumed = loaded_session is not None
    if loaded_session is None:
        system_prompt = build_system_prompt()
        config = _create_provider_config(chat_settings)
        loop = _create_agent_loop(
            settings=chat_settings,
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
    else:
        resume_settings = AgentletSettings(
            provider=_resolve_from_session(args, loaded_session, "provider_name"),
            model=_resolve_from_session(args, loaded_session, "model"),
            api_key=chat_settings.api_key,
            api_base=_resolve_from_session(args, loaded_session, "api_base"),
            temperature=_resolve_from_session(args, loaded_session, "temperature"),
            max_tokens=_resolve_from_session(args, loaded_session, "max_tokens"),
        )
        loop = _create_agent_loop(
            settings=resume_settings,
            provider_registry=provider_registry,
            tool_registry=tool_registry,
            system_prompt=loaded_session.context.system_prompt,
        )
        prompt = prompt_session or create_prompt_session(session_store.history_path)

    presenter = ChatPresenter(console or Console(file=stdout, stderr=False))
    return run_repl(
        loop=loop,
        prompt_input=prompt,
        presenter=presenter,
        session_store=session_store,
        cwd=working_dir,
        loaded_session=loaded_session,
        resumed=resumed,
    )


def _settings_from_args(args: Any, *, fallback: AgentletSettings) -> AgentletSettings:
    """Build effective chat settings from parsed CLI arguments."""
    def _value(name: str):
        value = getattr(args, name, None)
        return getattr(fallback, name) if value is None else value

    return AgentletSettings(
        provider=_value("provider"),
        model=_value("model"),
        api_key=_value("api_key"),
        api_base=_value("api_base"),
        temperature=_value("temperature"),
        max_tokens=_value("max_tokens"),
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
    has_resume_flags = bool(
        getattr(args, "continue_session", False)
        or getattr(args, "session_id", None)
        or getattr(args, "new_session", False)
    )

    if args.message is not None and has_resume_flags:
        raise ChatCLIError("Session flags cannot be combined with a one-shot message.")
    if args.print_mode and has_resume_flags:
        raise ChatCLIError("Session flags cannot be combined with --print.")

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

    if has_resume_flags and not is_tty:
        raise ChatCLIError("Session flags require an interactive TTY.")

    if not is_tty:
        return _read_and_validate_message(stdin), False

    return None, True


def _resolve_from_session(args: Any, loaded_session: LoadedSession, attr: str) -> Any:
    """Resolve a configuration value from the resumed transcript metadata."""
    return getattr(loaded_session.info, attr) if loaded_session.info else getattr(args, attr)

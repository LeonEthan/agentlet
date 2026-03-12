from __future__ import annotations

"""Mode selection and top-level chat command wiring."""

import asyncio
from pathlib import Path
from typing import Any, TextIO

from rich.console import Console

from agentlet.agent.agent_loop import AgentLoop, AgentTurnResult
from agentlet.agent.context import Context
from agentlet.agent.prompts.system_prompt import build_system_prompt
from agentlet.agent.providers.registry import ProviderConfig, ProviderRegistry
from agentlet.agent.tools.registry import ToolRegistry
from agentlet.cli.presenter import ChatPresenter
from agentlet.cli.prompt import build_prompt, create_prompt_session
from agentlet.cli.repl import run_repl
from agentlet.cli.sessions import LoadedSession, SessionError, SessionStore, load_session_for_resume


class ChatCLIError(ValueError):
    """Raised when CLI arguments resolve to an invalid chat mode."""


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
    system_prompt: str | None = None,
) -> AgentTurnResult:
    """Wire one-shot CLI arguments into runtime dependencies and execute one turn."""
    loop = _create_agent_loop(
        provider_name=provider_name,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        provider_registry=provider_registry,
        tool_registry=tool_registry,
        system_prompt=system_prompt,
    )
    return await loop.run_turn(message)


def _create_agent_loop(
    *,
    provider_name: str,
    model: str,
    api_key: str | None,
    api_base: str | None,
    temperature: float,
    max_tokens: int | None,
    provider_registry: ProviderRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    system_prompt: str | None = None,
) -> AgentLoop:
    """Build one configured runtime loop for the current CLI invocation."""
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
    return AgentLoop(
        provider=provider,
        tool_registry=tool_registry or ToolRegistry(),
        system_prompt=system_prompt or build_system_prompt(),
    )


def run_chat_command(
    args,
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
    session_store = SessionStore(working_dir)

    if not interactive:
        result = asyncio.run(
            run_chat(
                message=message or "",
                provider_name=args.provider,
                model=args.model,
                api_key=args.api_key,
                api_base=args.api_base,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                provider_registry=provider_registry,
                tool_registry=tool_registry,
            )
        )
        print(result.output, file=stdout)
        return 0

    try:
        loaded_session = load_session_for_resume(
            session_store,
            continue_session=args.continue_session,
            session_id=args.session_id,
        )
    except SessionError as exc:
        raise ChatCLIError(str(exc)) from exc

    if loaded_session is None:
        system_prompt = build_system_prompt()
        loop = _create_agent_loop(
            provider_name=args.provider,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            provider_registry=provider_registry,
            tool_registry=tool_registry,
            system_prompt=system_prompt,
        )
        prompt = prompt_session or create_prompt_session(session_store.history_path)
        info = session_store.start_session(
            provider_name=args.provider,
            model=args.model,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt=loop.system_prompt,
        )
        loaded_session = LoadedSession(
            info=info,
            context=Context(system_prompt=loop.system_prompt),
        )
    else:
        loop = _create_agent_loop(
            provider_name=_resolve_from_session(args, loaded_session, "provider_name"),
            model=_resolve_from_session(args, loaded_session, "model"),
            api_key=args.api_key,
            api_base=_resolve_from_session(args, loaded_session, "api_base"),
            temperature=_resolve_from_session(args, loaded_session, "temperature"),
            max_tokens=_resolve_from_session(args, loaded_session, "max_tokens"),
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
        continue_session=args.continue_session,
        session_id=args.session_id,
        loaded_session=loaded_session,
    )


def _read_and_validate_message(source: TextIO) -> str:
    """Read from source, strip whitespace, and validate non-empty."""
    message = source.read().strip()
    if not message:
        raise ChatCLIError("A message is required via argv or stdin.")
    return message


def _resolve_chat_mode(args, *, stdin: TextIO, stdin_isatty: bool | None = None) -> tuple[str | None, bool]:
    """Resolve CLI arguments and stdin shape into (message, interactive) tuple."""
    is_tty = stdin.isatty() if stdin_isatty is None else stdin_isatty
    has_resume_flags = bool(args.continue_session or args.session_id or args.new_session)

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


def _resolve_from_session(args, loaded_session: LoadedSession, attr: str) -> Any:
    """Resolve a configuration value from session info or CLI args."""
    return getattr(loaded_session.info, attr) if loaded_session.info else getattr(args, attr)

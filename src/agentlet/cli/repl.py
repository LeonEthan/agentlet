from __future__ import annotations

"""Interactive REPL orchestration for phase-2 chat sessions."""

import asyncio
import time
from pathlib import Path
from typing import Protocol

from agentlet.agent.agent_loop import AgentLoop
from agentlet.agent.context import Context
from agentlet.cli.commands import CommandError, command_help_lines, parse_command, summarize_history
from agentlet.cli.presenter import ChatPresenter
from agentlet.cli.sessions import LoadedSession, SessionStore, SessionTurnRecorder


class PromptInput(Protocol):
    """Small prompt protocol so smoke tests can inject a fake source."""

    def prompt(self, prompt_text: str | None = None) -> str: ...


def run_repl(
    *,
    loop: AgentLoop,
    prompt_input: PromptInput,
    presenter: ChatPresenter,
    session_store: SessionStore,
    cwd: Path,
    loaded_session: LoadedSession,
) -> int:
    """Run the interactive chat loop until the user exits."""
    current_session = loaded_session.info
    context = loaded_session.context

    presenter.show_session_header(
        session_id=current_session.session_id,
        provider_name=current_session.provider_name,
        model=current_session.model,
        cwd=cwd,
    )

    last_idle_interrupt = 0.0

    while True:
        try:
            raw_input = prompt_input.prompt()
            last_idle_interrupt = 0.0
        except KeyboardInterrupt:
            now = time.monotonic()
            if now - last_idle_interrupt <= 2.0:
                presenter.show_notice("Exiting interactive session.")
                return 0
            last_idle_interrupt = now
            presenter.show_notice("Input cleared. Press Ctrl+C again within 2s to exit.")
            continue
        except EOFError:
            presenter.show_notice("Session closed.")
            return 0

        message = raw_input.strip()
        if not message:
            continue

        try:
            command = parse_command(message)
        except CommandError as exc:
            presenter.show_error("Command error", str(exc))
            continue

        if command is not None:
            if command == "help":
                presenter.show_help(command_help_lines())
                continue
            if command == "status":
                presenter.show_status(
                    session_id=current_session.session_id,
                    provider_name=current_session.provider_name,
                    model=current_session.model,
                    cwd=cwd,
                    message_count=len(context.history),
                    tool_names=loop.tool_registry.get_tool_names(),
                )
                continue
            if command == "history":
                presenter.show_history(summarize_history(context.history))
                continue
            if command == "clear":
                presenter.clear()
                continue
            if command == "new":
                current_session = session_store.start_session(
                    provider_name=current_session.provider_name,
                    model=current_session.model,
                    api_base=current_session.api_base,
                    temperature=current_session.temperature,
                    max_tokens=current_session.max_tokens,
                    system_prompt=loop.system_prompt,
                )
                context = Context(system_prompt=loop.system_prompt)
                presenter.show_session_header(
                    session_id=current_session.session_id,
                    provider_name=current_session.provider_name,
                    model=current_session.model,
                    cwd=cwd,
                )
                continue
            if command == "exit":
                presenter.show_notice("Session closed.")
                return 0

        recorder = SessionTurnRecorder()

        def handle_event(event) -> None:
            recorder.observe(event)
            presenter.handle_event(event)

        try:
            asyncio.run(
                loop.run_turn(
                    message,
                    context=context,
                    event_sink=handle_event,
                    stream=True,
                )
            )
        except KeyboardInterrupt:
            presenter.stop_stream()
            presenter.show_notice("Turn cancelled.")
            continue
        except Exception as exc:
            presenter.stop_stream()
            presenter.show_error("Turn failed", str(exc))
            continue

        records = recorder.build_records(current_session.session_id)
        session_store.append_records(
            current_session,
            records,
            update_latest=True,
        )

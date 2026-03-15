from __future__ import annotations

"""Slash command parsing and local transcript summaries."""

from dataclasses import dataclass
from enum import Enum

from agentlet.agent.context import Message


class CommandName(str, Enum):
    """Interactive slash command names."""

    HELP = "help"
    EXIT = "exit"
    STATUS = "status"
    NEW = "new"
    HISTORY = "history"
    CLEAR = "clear"


# Set of valid command values for fast lookup
_COMMANDS: set[str] = {cmd.value for cmd in CommandName}


class CommandError(ValueError):
    """Raised when interactive slash command input is invalid."""


@dataclass(frozen=True)
class TurnSummary:
    """One displayed user/assistant turn with its original turn number."""

    number: int
    user_text: str
    assistant_text: str


def parse_command(raw_input: str) -> CommandName | None:
    """Parse a slash command or return None for regular user text."""
    if not raw_input.startswith("/"):
        return None

    token = raw_input.strip().split()[0][1:]
    if token not in _COMMANDS:
        raise CommandError(f"Unknown command: /{token}")
    if raw_input.strip() != f"/{token}":
        raise CommandError(f"Command /{token} does not take arguments.")
    return CommandName(token)


def command_help_lines() -> list[str]:
    """Return the short interactive help text."""
    return [
        "/help    show interactive commands",
        "/status  show current session details",
        "/history show recent turn summaries",
        "/new     start a fresh session",
        "/clear   clear the visible terminal",
        "/exit    leave the session",
        "Enter submits, Alt+Enter inserts a newline.",
    ]


def summarize_history(history: list[Message], *, limit: int = 10) -> list[TurnSummary]:
    """Summarize recent user/assistant turns for the interactive shell."""
    turns: list[TurnSummary] = []
    pending_user: str | None = None
    pending_assistant = ""
    turn_number = 0

    for message in history:
        if message.role == "user":
            if pending_user is not None:
                turn_number += 1
                turns.append(
                    TurnSummary(turn_number, pending_user, pending_assistant)
                )
            pending_user = message.content or ""
            pending_assistant = ""
            continue
        if message.role == "assistant" and pending_user is not None:
            pending_assistant = message.content or ""

    if pending_user is not None:
        turn_number += 1
        turns.append(TurnSummary(turn_number, pending_user, pending_assistant))

    return turns[-limit:]

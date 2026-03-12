from __future__ import annotations

"""Prompt building utilities and prompt_toolkit helpers for the interactive shell."""

import sys
from pathlib import Path
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings


def create_prompt_session(history_path: Path) -> PromptSession[str]:
    """Build one prompt session with local history and multiline key bindings."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        key_bindings=_build_key_bindings(),
    )


def _build_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    def _newline(event) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def build_prompt(inputs: list[str] | None, file: TextIO | None) -> str | None:
    """
    Build prompt from command-line arguments or stdin.

    Priority:
    1. Explicit inputs (CLI arguments)
    2. File input (--file)
    3. Stdin (if not a tty)
    4. Interactive mode (returns None)
    """
    # Explicit inputs
    if inputs:
        return " ".join(inputs)

    # File input
    if file:
        return file.read()

    # Check if stdin has data
    if not sys.stdin.isatty():
        return sys.stdin.read()

    # Interactive mode
    return None


__all__ = ["create_prompt_session", "build_prompt"]

from __future__ import annotations

"""Interactive approval helpers for unsafe tool execution."""

from dataclasses import dataclass, field
from typing import Awaitable
from typing import Protocol, TextIO

from agentlet.agent.tools.registry import ToolApprovalRequest


class ApprovalPromptClosed(EOFError):
    """Raised when the user closes an interactive approval prompt."""


class ApprovalPrompt(Protocol):
    """Prompt interface shared by prompt_toolkit sessions and test doubles."""

    def prompt(self, prompt_text: str | None = None) -> str: ...


class AsyncApprovalPrompt(Protocol):
    """Async prompt interface exposed by prompt_toolkit sessions."""

    def prompt_async(
        self, prompt_text: str | None = None
    ) -> Awaitable[str]: ...


@dataclass
class InteractiveApprovalHandler:
    """Session-scoped approval handler for write, bash, and network actions."""

    prompt_input: ApprovalPrompt | None = None
    stdin: TextIO | None = None
    stdout: TextIO | None = None
    auto_approve: bool = False
    approved_scopes: set[str] = field(default_factory=set)
    _tty_file: TextIO | None = field(default=None, repr=False)
    _tty_attempted: bool = field(default=False, repr=False)

    def __enter__(self) -> InteractiveApprovalHandler:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close any opened file handles."""
        if self._tty_file is not None:
            self._tty_file.close()
            self._tty_file = None

    async def approve(self, request: ToolApprovalRequest) -> bool:
        if self.auto_approve or request.scope in self.approved_scopes:
            return True
        if not self.can_prompt():
            return False

        prompt_text = (
            f"Approve {request.summary}? "
            "[y]es/[n]o/[a]ll-for-session: "
        )
        while True:
            try:
                response = (await self._prompt(prompt_text)).strip().lower()
            except EOFError as exc:
                raise ApprovalPromptClosed() from exc
            if response in {"y", "yes"}:
                return True
            if response in {"n", "no", ""}:
                return False
            if response in {"a", "all"}:
                self.approved_scopes.add(request.scope)
                return True

    def can_prompt(self) -> bool:
        if self.prompt_input is not None:
            return True
        if self.stdin is None or self.stdout is None:
            return False
        stdin_isatty = getattr(self.stdin, "isatty", None)
        stdout_isatty = getattr(self.stdout, "isatty", None)
        is_tty = bool(
            stdin_isatty is not None
            and stdin_isatty()
            and stdout_isatty is not None
            and stdout_isatty()
        )
        if is_tty:
            return True
        # Check if we can open the controlling terminal as a fallback
        if self._tty_file is None and not self._tty_attempted:
            self._tty_attempted = True
            try:
                tty_file = open("/dev/tty", "r+")
                # Verify it's actually a TTY before using it
                if tty_file.isatty():
                    self._tty_file = tty_file
                else:
                    tty_file.close()
            except (OSError, PermissionError):
                pass
        return self._tty_file is not None

    async def _prompt(self, prompt_text: str) -> str:
        if self.prompt_input is not None:
            prompt_async = getattr(self.prompt_input, "prompt_async", None)
            original_message = _get_prompt_message(self.prompt_input)
            try:
                if callable(prompt_async):
                    return await prompt_async(prompt_text)
                return self.prompt_input.prompt(prompt_text)
            finally:
                _restore_prompt_message(self.prompt_input, original_message)
        # Use the controlling terminal if available (e.g., when stdin is piped)
        if self._tty_file is not None:
            self._tty_file.write(prompt_text)
            self._tty_file.flush()
            return self._tty_file.readline()
        assert self.stdin is not None
        assert self.stdout is not None
        self.stdout.write(prompt_text)
        self.stdout.flush()
        return self.stdin.readline()


def _get_prompt_message(prompt_input: object) -> object:
    """Capture the current prompt message when using PromptSession-like inputs."""
    return getattr(prompt_input, "message", None)


def _restore_prompt_message(prompt_input: object, message: object) -> None:
    """Restore the original prompt message after an approval prompt overrides it."""
    if not hasattr(prompt_input, "message"):
        return
    try:
        setattr(prompt_input, "message", message)
    except (AttributeError, TypeError):
        return

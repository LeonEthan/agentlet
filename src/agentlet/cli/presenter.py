from __future__ import annotations

"""Rich-based rendering for the phase-2 CLI."""

from pathlib import Path
from textwrap import shorten

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown

from agentlet.agent.agent_loop import TurnEvent


# Layout constants
_PATH_MAX_LEN = 40
_PATH_HEAD_LEN = 15
_PATH_TAIL_LEN = 20
_SEPARATOR_WIDTH = 40
_HELP_CMD_WIDTH = 12
_STATUS_LABEL_WIDTH = 10
_TOOL_ARGS_LIMIT = 40
_REFRESH_RATE = 12
_HISTORY_LIMIT = 20

SEPARATOR = "─" * _SEPARATOR_WIDTH


class Theme:
    """Color palette and styling for the TUI."""

    DIM = "#6B7280"  # Secondary text, hints
    SUCCESS = "#059669"  # Success states
    ERROR = "#DC2626"  # Errors


STATUS_ICONS = {
    "pending": "⠋",  # Braille spinner
    "success": "✓",  # Checkmark
    "error": "✗",  # X mark
}


class ChatPresenter:
    """Render session state, turn events, and CLI messages."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._assistant_live: Live | None = None
        self._assistant_text = ""

    def show_session_header(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
    ) -> None:
        """Display a single-line session header with truncated path."""
        path_str = str(cwd)
        if len(path_str) > _PATH_MAX_LEN:
            path_str = (
                path_str[:_PATH_HEAD_LEN] + "..." + path_str[-_PATH_TAIL_LEN:]
            )
        line = f"agentlet · {model} · {path_str} · /help for commands"
        self.console.print(line, style=Theme.DIM)

    def show_help(self, lines: list[str]) -> None:
        """Display help text with command/description pairs aligned."""
        self.console.print("Commands:")
        for line in lines:
            if line.startswith("/"):
                cmd, _sep, desc = line.partition(" ")
                self.console.print(
                    f"  {cmd:<{_HELP_CMD_WIDTH}} {desc.strip()}", style=Theme.DIM
                )
            else:
                self.console.print(f"  {line}")

    def show_status(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
        message_count: int,
        tool_names: list[str],
    ) -> None:
        """Display session status as aligned key-value pairs."""
        label_width = _STATUS_LABEL_WIDTH
        tools = ", ".join(tool_names) if tool_names else "(none)"
        lines = [
            f"{'Session:':<{label_width}} {session_id}",
            f"{'Provider:':<{label_width}} {provider_name}",
            f"{'Model:':<{label_width}} {model}",
            f"{'CWD:':<{label_width}} {cwd}",
            f"{'Messages:':<{label_width}} {message_count}",
            f"{'Tools:':<{label_width}} {tools}",
        ]
        self.console.print("\n".join(lines))

    def show_history(
        self, turns: list[tuple[str, str]], *, limit: int = _HISTORY_LIMIT
    ) -> None:
        """Display conversation history with turn separators."""
        if not turns:
            self.console.print("No completed turns yet.", style=Theme.DIM)
            return

        output: list[str] = []
        excess = len(turns) - limit
        if excess > 0:
            display_turns = turns[-limit:]
        else:
            display_turns = turns

        for i, (user_text, assistant_text) in enumerate(display_turns, 1):
            output.append(f"Turn {i}")
            output.append(SEPARATOR)
            output.append(f"› {_truncate(user_text, 80)}")
            output.append("")
            output.append(_truncate(assistant_text, 80) if assistant_text else "")
            output.append("")

        if excess > 0:
            output.append(f"... ({excess} earlier turns hidden)")

        self.console.print("\n".join(output), style=Theme.DIM)

    def show_notice(self, message: str) -> None:
        """Display a notice message."""
        self.console.print(f"⚠ {message}", style=Theme.DIM)

    def show_error(self, title: str, message: str) -> None:
        """Display an error message and stop any active stream."""
        self.stop_stream()
        self.console.print(
            f"{STATUS_ICONS['error']} {title}: {message}", style=Theme.ERROR
        )

    def clear(self) -> None:
        """Clear the console."""
        self.console.clear()

    def handle_event(self, event: TurnEvent) -> None:
        """Process a turn event and update the display."""
        if event.kind == "assistant_delta" and event.text:
            self._assistant_text += event.text
            self._ensure_stream().update(self._render_stream())
            return

        if event.kind == "assistant_completed":
            self._finish_assistant_stream(event.content)
            return

        if event.kind == "tool_started" and event.tool_call is not None:
            self.stop_stream()
            args = _truncate(event.tool_call.arguments_json, _TOOL_ARGS_LIMIT)
            self.console.print(
                f"{STATUS_ICONS['pending']} {event.tool_call.name}({args})...",
                style=Theme.DIM,
            )
            return

        if event.kind == "tool_completed" and event.tool_result is not None:
            self.stop_stream()
            self.console.print(
                f"{STATUS_ICONS['success']} {event.tool_result.name} completed",
                style=Theme.SUCCESS,
            )

    def stop_stream(self) -> None:
        """Stop the active assistant stream if one exists."""
        if self._assistant_live is not None:
            self._assistant_live.update(self._render_stream())
            self._assistant_live.stop()
            self._assistant_live = None
            self._assistant_text = ""
            self.console.print()

    def _finish_assistant_stream(self, final_content: str | None) -> None:
        """Finalize the assistant stream with optional final content."""
        if self._assistant_live is not None:
            if final_content is not None:
                self._assistant_text = final_content
            self._assistant_live.update(self._render_stream())
            self._assistant_live.stop()
            self._assistant_live = None
            self._assistant_text = ""
            self.console.print()
            return

        if final_content:
            self.console.print(Markdown(final_content))
            self.console.print()

    def _ensure_stream(self) -> Live:
        """Ensure a live stream exists and is started."""
        if self._assistant_live is None:
            self._assistant_live = Live(
                self._render_stream(),
                console=self.console,
                refresh_per_second=_REFRESH_RATE,
                transient=False,
            )
            self._assistant_live.start()
        return self._assistant_live

    def _render_stream(self) -> Markdown:
        """Render the current assistant stream content."""
        content = self._assistant_text or " "
        return Markdown(content)


def _truncate(text: str | None, limit: int = 80) -> str:
    """Truncate text to the specified limit with ellipsis."""
    return shorten(text or "", width=limit, placeholder="...")

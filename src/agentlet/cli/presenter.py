from __future__ import annotations

"""Rich-based rendering for the phase-2 CLI."""

import json
from pathlib import Path
from textwrap import shorten

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape

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
            self.console.print(
                f"{STATUS_ICONS['pending']} "
                f"{_format_tool_call(event.tool_call.name, event.tool_call.arguments_json)}...",
                style=Theme.DIM,
            )
            return

        if event.kind == "tool_completed" and event.tool_result is not None:
            self.stop_stream()
            self.console.print(
                f"{STATUS_ICONS['success']} "
                f"{escape(_format_tool_result(event.tool_result.name, event.tool_result.content))}",
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
    if not text or limit <= 0:
        return ""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized

    placeholder = "..."
    if limit <= len(placeholder):
        return placeholder[:limit]

    if not any(char.isspace() for char in normalized):
        return _truncate_middle(normalized, limit, placeholder)

    shortened = shorten(normalized, width=limit, placeholder=placeholder)
    if shortened != placeholder:
        return shortened

    return _truncate_middle(normalized, limit, placeholder)


def _truncate_middle(text: str, limit: int, placeholder: str) -> str:
    """Preserve both ends of a long token-like value."""
    keep = limit - len(placeholder)
    head = (keep + 1) // 2
    tail = keep - head
    if tail <= 0:
        return text[:head] + placeholder
    return f"{text[:head]}{placeholder}{text[-tail:]}"


def _format_tool_call(name: str, arguments_json: str) -> str:
    """Render a compact, high-signal tool start line."""
    arguments = _load_json_object(arguments_json)
    if arguments is None:
        return f"{name}({_truncate(arguments_json, _TOOL_ARGS_LIMIT)})"

    keys_by_tool = {
        "read": ("path", "start_line", "end_line"),
        "write": ("path",),
        "edit": ("path",),
        "bash": ("command",),
        "glob": ("pattern",),
        "grep": ("pattern", "glob"),
        "web_search": ("query",),
        "web_fetch": ("url",),
    }
    keys = keys_by_tool.get(name, tuple(arguments.keys()))
    parts = []
    for key in keys:
        if key not in arguments:
            continue
        value = arguments[key]
        rendered = repr(value) if isinstance(value, str) else str(value)
        parts.append(f"{key}={_truncate(rendered, _TOOL_ARGS_LIMIT)}")
    if not parts:
        return name
    return f"{name}({', '.join(parts)})"


def _format_tool_result(name: str, content: str) -> str:
    """Render a compact, tool-specific success summary."""
    payload = _load_json_object(content)
    if payload is None:
        return f"{name} completed"

    if name == "read":
        path = payload.get("path", "")
        start_line = payload.get("start_line")
        end_line = payload.get("end_line")
        truncated = payload.get("truncated", False)
        return (
            f"read {_truncate(str(path), 48)} "
            f"lines {start_line}-{end_line} truncated={truncated}"
        )
    if name in {"write", "edit"}:
        path = payload.get("path", "")
        if name == "edit":
            replacements = payload.get("total_replacements", 0)
            return f"edit {_truncate(str(path), 48)} replacements={replacements}"
        return f"write {_truncate(str(path), 48)} created"
    if name == "bash":
        command = _truncate(str(payload.get("command", "")), 48)
        exit_code = payload.get("exit_code")
        return f"bash {command} exit={exit_code}"
    if name in {"glob", "grep"}:
        match_count = len(payload.get("matches", []))
        return f"{name} matches={match_count}"
    if name == "web_search":
        result_count = len(payload.get("results", []))
        return f"web_search results={result_count}"
    if name == "web_fetch":
        final_url = _truncate(str(payload.get("final_url", payload.get("url", ""))), 48)
        title = payload.get("title")
        truncated = payload.get("truncated", False)
        if title:
            return (
                f"web_fetch {final_url} "
                f"title={_truncate(str(title), 32)} truncated={truncated}"
            )
        return f"web_fetch {final_url} truncated={truncated}"
    return f"{name} completed"


def _load_json_object(raw: str) -> dict[str, object] | None:
    """Parse a JSON object payload when possible."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None

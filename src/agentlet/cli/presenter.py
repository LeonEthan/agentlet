from __future__ import annotations

"""Rich-based rendering for the phase-2 CLI."""

from pathlib import Path
from textwrap import shorten

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from agentlet.agent.agent_loop import TurnEvent


class ChatPresenter:
    """Render session state, turn events, and CLI messages."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self._assistant_live: Live | None = None
        self._assistant_text = ""

    def _build_info_table(self, rows: list[tuple[str, str]], title: str, style: str) -> None:
        """Build and print a grid table with key-value rows."""
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_column(justify="left")
        for key, value in rows:
            table.add_row(key, value)
        self.console.print(Panel(table, title=title, border_style=style))

    def show_session_header(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
        resumed: bool,
    ) -> None:
        status = "resumed" if resumed else "new"
        rows = [
            ("provider", provider_name),
            ("model", model),
            ("cwd", str(cwd)),
            ("session", session_id),
            ("mode", status),
            ("hint", "/help for commands"),
        ]
        self._build_info_table(rows, title="agentlet chat", style="blue")

    def show_help(self, lines: list[str]) -> None:
        body = "\n".join(lines)
        self.console.print(Panel(body, title="commands", border_style="blue"))

    def show_status(
        self,
        *,
        session_id: str,
        provider_name: str,
        model: str,
        cwd: Path,
        message_count: int,
    ) -> None:
        rows = [
            ("session", session_id),
            ("provider", provider_name),
            ("model", model),
            ("cwd", str(cwd)),
            ("messages", str(message_count)),
        ]
        self._build_info_table(rows, title="status", style="cyan")

    def show_history(self, turns: list[tuple[str, str]]) -> None:
        if not turns:
            self.console.print(Panel("No completed turns yet.", title="history"))
            return

        table = Table(title="recent turns")
        table.add_column("user")
        table.add_column("assistant")
        for user_text, assistant_text in turns:
            table.add_row(_truncate(user_text), _truncate(assistant_text))
        self.console.print(table)

    def show_notice(self, message: str) -> None:
        self.console.print(Panel(message, border_style="yellow"))

    def show_error(self, title: str, message: str) -> None:
        self.stop_stream()
        self.console.print(Panel(message, title=title, border_style="red"))

    def clear(self) -> None:
        self.console.clear()

    def handle_event(self, event: TurnEvent) -> None:
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
                f"[cyan]tool start[/] {event.tool_call.name}({_truncate(event.tool_call.arguments_json, 60)})"
            )
            return

        if event.kind == "tool_completed" and event.tool_result is not None:
            self.stop_stream()
            self.console.print(
                f"[green]tool done[/] {event.tool_result.name}: {_truncate(event.tool_result.content)}"
            )

    def stop_stream(self) -> None:
        if self._assistant_live is not None:
            self._assistant_live.update(self._render_stream())
            self._assistant_live.stop()
            self._assistant_live = None
            self._assistant_text = ""
            self.console.print()

    def _finish_assistant_stream(self, final_content: str | None) -> None:
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
        if self._assistant_live is None:
            self._assistant_live = Live(
                self._render_stream(),
                console=self.console,
                refresh_per_second=12,
                transient=False,
            )
            self._assistant_live.start()
        return self._assistant_live

    def _render_stream(self) -> Group:
        content = self._assistant_text or " "
        return Group(Rule("assistant"), Markdown(content))


def _truncate(text: str | None, limit: int = 80) -> str:
    return shorten(text or "", width=limit, placeholder="...")

"""Rich-based terminal UI components for the agentlet CLI."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, TextIO

from rich.align import Align
from rich.box import ROUNDED
from rich.console import Console, Group, RenderableType
from rich.json import JSON as RichJSON
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.spinner import Spinner
from rich.status import Status
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from agentlet.core.interrupts import (
        ApprovalRequest,
        UserQuestionRequest,
    )

# Claude Code-inspired color scheme - refined
CLI_THEME = Theme(
    {
        # Primary colors
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red",
        # Tool colors
        "tool_name": "bright_magenta",
        "tool_border": "magenta",
        "tool_arg": "bright_cyan",
        "tool_output": "white",
        "tool_error": "red",
        # UI elements
        "prompt": "bright_blue",
        "user_input": "bright_white",
        "assistant": "bright_green",
        "system": "dim white",
        "separator": "dim",
        # Approval/Question colors
        "approval_pending": "yellow",
        "approved": "green",
        "rejected": "red",
        "question": "bright_cyan",
        # Metadata
        "muted": "dim",
        "timestamp": "dim cyan",
        # New: Thinking/processing states
        "thinking": "bright_blue",
        "processing": "bright_yellow",
    }
)

# Tool icons mapping - expanded
TOOL_ICONS = {
    "Read": "📄",
    "Write": "✏️",
    "Edit": "📝",
    "Bash": "⚡",
    "Glob": "🔍",
    "Grep": "🔎",
    "WebSearch": "🌐",
    "WebFetch": "📥",
    "AskUserQuestion": "❓",
}

# Tool category descriptions
TOOL_CATEGORIES = {
    "Read": "reading file",
    "Write": "writing file",
    "Edit": "editing file",
    "Bash": "running command",
    "Glob": "searching files",
    "Grep": "searching content",
    "WebSearch": "searching web",
    "WebFetch": "fetching URL",
    "AskUserQuestion": "asking question",
}

# Approval category styling
APPROVAL_STYLES = {
    "read_only": ("blue", "👁️"),
    "mutating": ("yellow", "⚠️"),
    "exec": ("red", "🔒"),
    "external_or_interrupt": ("magenta", "🌐"),
}


class ThinkingIndicator:
    """Live indicator showing the model is thinking/processing."""

    def __init__(self, console: Console, message: str = "Thinking...") -> None:
        self.console = console
        self.message = message
        self._live: Live | None = None
        self._start_time: float = 0.0

    def __enter__(self) -> ThinkingIndicator:
        self._start_time = time.time()
        spinner = Spinner(
            "dots2",
            text=Text(self.message, style="thinking"),
        )
        self._live = Live(
            spinner,
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            elapsed = time.time() - self._start_time
            self._live.__exit__(*args)
            self._live = None
            # Show completion with elapsed time if > 1s
            if elapsed > 1.0:
                self.console.print(
                    Text.assemble(
                        ("✓ ", "success"),
                        ("Done", "success"),
                        (f" ({elapsed:.1f}s)", "timestamp"),
                    ),
                    style="dim",
                )


class ToolExecutionDisplay:
    """Display for tool execution with live status updates."""

    def __init__(self, console: Console, tool_name: str, arguments: dict) -> None:
        self.console = console
        self.tool_name = tool_name
        self.arguments = arguments
        self._live: Live | None = None
        self._start_time: float = 0.0
        self._icon = TOOL_ICONS.get(tool_name, "🔧")
        self._category = TOOL_CATEGORIES.get(tool_name, "processing")

    def __enter__(self) -> ToolExecutionDisplay:
        self._start_time = time.time()

        # Build the initial display with arguments
        content_parts = self._build_content(running=True)

        self._live = Live(
            Group(*content_parts),
            console=self.console,
            refresh_per_second=12,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def _build_content(self, running: bool = False) -> list[RenderableType]:
        """Build the display content."""
        elapsed = time.time() - self._start_time

        # Header with icon and status
        status_icon = "⏳" if running else "✓"
        status_style = "processing" if running else "success"

        header_text = Text.assemble(
            (self._icon, "info"),
            " ",
            (self.tool_name, "tool_name bold"),
            " ",
            (f"({self._category})", "muted"),
            " ",
            (status_icon, status_style),
            (f" {elapsed:.1f}s" if elapsed > 0.5 else "", "timestamp"),
        )

        content_parts: list[RenderableType] = []

        # Arguments section
        if self.arguments:
            json_str = json.dumps(self.arguments, indent=2, ensure_ascii=False)
            syntax = Syntax(
                json_str,
                "json",
                theme="monokai",
                background_color="default",
                line_numbers=False,
            )
            content_parts.append(syntax)

        panel = Panel(
            Group(*content_parts) if content_parts else Text("No arguments", style="muted"),
            title=header_text,
            border_style="tool_border" if running else "success",
            box=ROUNDED,
            padding=(0, 1),
        )

        return [panel]

    def update_result(self, output: str, is_error: bool = False) -> None:
        """Update the display with the result."""
        elapsed_ms = (time.time() - self._start_time) * 1000

        status_icon = "❌" if is_error else "✓"
        status_style = "error" if is_error else "success"

        header_text = Text.assemble(
            (self._icon, "info"),
            " ",
            (self.tool_name, "tool_name bold"),
            " ",
            (status_icon, status_style),
            " ",
            (f"({elapsed_ms:.0f}ms)", "timestamp"),
        )

        # Format output
        if is_error:
            content: Text | Syntax = Text(output, style="error")
        else:
            content = format_code_output(output)

        panel = Panel(
            content,
            title=header_text,
            border_style="red" if is_error else "green",
            box=ROUNDED,
            padding=(0, 1),
        )

        if self._live:
            self._live.update(panel)

    def finalize(self, output: str, is_error: bool = False) -> None:
        """Finalize the display and print the result."""
        if self._live:
            self._live.stop()
            self._live = None

        elapsed_ms = (time.time() - self._start_time) * 1000

        status_icon = "❌" if is_error else "✓"
        status_style = "error" if is_error else "success"

        header_text = Text.assemble(
            (self._icon, "info"),
            " ",
            (self.tool_name, "tool_name bold"),
            " ",
            (status_icon, status_style),
            " ",
            (f"({elapsed_ms:.0f}ms)", "timestamp"),
        )

        # Format output
        if is_error:
            content: Text | Syntax = Text(output, style="error")
        else:
            content = format_code_output(output)

        panel = Panel(
            content,
            title=header_text,
            border_style="red" if is_error else "green",
            box=ROUNDED,
            padding=(0, 1),
        )

        self.console.print(panel)

# Tool icons mapping
TOOL_ICONS = {
    "Read": "📄",
    "Write": "✏️",
    "Edit": "📝",
    "Bash": "⚡",
    "Glob": "🔍",
    "Grep": "🔎",
    "WebSearch": "🌐",
    "WebFetch": "📥",
    "AskUserQuestion": "❓",
}

# Approval category styling
APPROVAL_STYLES = {
    "read_only": ("blue", "👁️"),
    "mutating": ("yellow", "⚠️"),
    "exec": ("red", "🔒"),
    "external_or_interrupt": ("magenta", "🌐"),
}


def create_console(file: TextIO | None = None) -> Console:
    """Create a rich console with the agentlet theme."""
    return Console(
        theme=CLI_THEME,
        soft_wrap=True,
        highlight=True,
        file=file,
    )


class StreamingResponse:
    """Handle streaming response display with live updates."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.content = ""
        self.live: Live | None = None
        self._start_time = time.time()

    def __enter__(self) -> StreamingResponse:
        self._start_time = time.time()
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=20,
            transient=False,
            vertical_overflow="visible",
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self.live:
            # Clear the live display and stop it
            self.live.stop()
            self.live = None
            # Print final content without cursor on a fresh line
            if self.content:
                self.console.print(Markdown(self.content))

    def _render(self) -> RenderableType:
        """Render the current content with a cursor."""
        # Show content with a blinking cursor indicator
        elapsed = time.time() - self._start_time
        cursor = "▌" if int(elapsed * 2) % 2 == 0 else " "
        return Markdown(self.content + cursor)

    def append(self, text: str) -> None:
        """Append text to the streaming content."""
        self.content += text
        if self.live:
            self.live.update(self._render())

    def finalize(self) -> None:
        """Finalize the streaming display."""
        if self.live:
            self.live.stop()
            self.live = None
            if self.content:
                self.console.print(Markdown(self.content))


def format_tool_name(name: str) -> Text:
    """Format a tool name with icon and styling."""
    icon = TOOL_ICONS.get(name, "🔧")
    return Text.assemble(
        (icon, "info"),
        " ",
        (name, "tool_name bold"),
    )


def format_approval_request(request: ApprovalRequest) -> Panel:
    """Format an approval request as a rich panel."""
    category_style, category_icon = APPROVAL_STYLES.get(
        request.approval_category, ("yellow", "⚠️")
    )

    # Build header
    header = Text.assemble(
        (category_icon, category_style),
        " ",
        ("Approval Required", f"{category_style} bold"),
        " ",
        (f"[{request.approval_category}]", f"{category_style} dim"),
    )

    # Build content
    content_parts: list[Text | Syntax | RichJSON] = []

    # Prompt
    content_parts.append(Text(request.prompt, style="bright_white"))
    content_parts.append(Text(""))

    # Arguments
    if request.arguments:
        content_parts.append(Text("Arguments:", style="muted"))
        json_str = json.dumps(request.arguments, indent=2, ensure_ascii=False)
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai",
            background_color="default",
        )
        content_parts.append(syntax)

    # Footer with instructions
    footer = Text.assemble(
        ("Approve? ", "prompt"),
        ("[y/N]", "muted"),
    )
    content_parts.append(Text(""))
    content_parts.append(footer)

    return Panel(
        Group(*content_parts),
        title=header,
        border_style=category_style,
        box=ROUNDED,
        padding=(0, 1),
    )


def format_question_interrupt(request: UserQuestionRequest) -> Panel:
    """Format a question interrupt as a rich panel."""
    header = Text.assemble(
        ("❓", "question"),
        " ",
        ("Agent Question", "question bold"),
    )

    content_parts: list[Text | Table] = []

    # Prompt
    content_parts.append(Text(request.prompt, style="bright_white"))
    content_parts.append(Text(""))

    # Options
    if request.options:
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
        )
        table.add_column("index", style="muted", justify="right")
        table.add_column("label", style="bright_cyan")
        table.add_column("value", style="muted")

        for idx, option in enumerate(request.options, start=1):
            table.add_row(
                f"{idx}.",
                option.label,
                f"[{option.value}]",
            )
        content_parts.append(table)

    if request.allow_free_text:
        content_parts.append(
            Text("(Free-text answers are also allowed)", style="muted italic")
        )

    content_parts.append(Text(""))
    content_parts.append(Text.assemble(("Answer: ", "prompt")))

    return Panel(
        Group(*content_parts),
        title=header,
        border_style="bright_cyan",
        box=ROUNDED,
        padding=(0, 1),
    )


def format_tool_output(
    tool_name: str,
    output: str,
    is_error: bool = False,
    execution_time_ms: float | None = None,
) -> Panel:
    """Format tool output as a rich panel."""
    icon = TOOL_ICONS.get(tool_name, "🔧")
    status_icon = "❌" if is_error else "✓"
    status_style = "error" if is_error else "success"

    header_parts = [
        (icon, "info"),
        " ",
        (tool_name, "tool_name"),
        " ",
        (status_icon, status_style),
    ]

    if execution_time_ms is not None:
        header_parts.extend([
            " ",
            (f"({execution_time_ms:.0f}ms)", "timestamp"),
        ])

    header = Text.assemble(*header_parts)

    # Format output content
    if is_error:
        content: Text | Syntax = Text(output, style="error")
    else:
        # Try to detect and format code/json
        content = format_code_output(output)

    return Panel(
        content,
        title=header,
        border_style="red" if is_error else "magenta",
        box=ROUNDED,
        padding=(0, 1),
    )


def format_code_output(output: str, language: str | None = None) -> Text | Syntax:
    """Detect and format code output appropriately.

    Args:
        output: The code/output to format
        language: Optional language hint for syntax highlighting
    """
    stripped = output.strip()

    # Use provided language or auto-detect
    detected_lang = language

    if detected_lang is None:
        # Try JSON
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                detected_lang = "json"
            except json.JSONDecodeError:
                pass

        # Try Python
        if detected_lang is None and any(
            stripped.startswith(keyword)
            for keyword in ("def ", "class ", "import ", "from ", "if ", "for ", "while ")
        ):
            detected_lang = "python"

        # Try TypeScript/JavaScript
        if detected_lang is None and any(
            stripped.startswith(keyword)
            for keyword in ("const ", "let ", "var ", "function ", "export ", "import ")
        ):
            detected_lang = "typescript"

        # Try Shell/Bash
        if detected_lang is None and any(
            stripped.startswith(prefix)
            for prefix in ("#!/bin/bash", "#!/bin/sh", "echo ", "ls ", "cat ", "grep ")
        ):
            detected_lang = "bash"

    if detected_lang:
        return Syntax(
            output,
            detected_lang,
            theme="monokai",
            background_color="default",
            line_numbers=True,
            word_wrap=True,
        )

    # Default to plain text with wrapping
    return Text(output)


def format_assistant_message(content: str) -> Markdown:
    """Format an assistant message with markdown rendering."""
    return Markdown(
        content,
        code_theme="monokai",
    )


def format_system_message(message: str, style: str = "info") -> Text:
    """Format a system message."""
    return Text(f"ℹ {message}", style=f"{style} dim")


@contextmanager
def tool_execution_spinner(console: Console, tool_name: str) -> Iterator[None]:
    """Context manager for showing a spinner during tool execution."""
    icon = TOOL_ICONS.get(tool_name, "🔧")
    spinner = Spinner(
        "dots",
        text=Text.assemble(
            (icon, "info"),
            " ",
            (tool_name, "tool_name"),
            " ",
            ("...", "muted"),
        ),
    )

    with Live(
        spinner,
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        yield


class StreamingMarkdown:
    """Handler for streaming markdown content with live rendering."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.content = ""
        self.live: Live | None = None

    def __enter__(self) -> StreamingMarkdown:
        self.live = Live(
            console=self.console,
            refresh_per_second=16,
            transient=False,
        )
        self.live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self.live:
            self.live.__exit__(*args)
            self.live = None

    def append(self, text: str) -> None:
        """Append text to the streaming content."""
        self.content += text
        if self.live:
            self.live.update(Markdown(self.content))

    def finalize(self) -> None:
        """Finalize the streaming display."""
        if self.live:
            self.live.stop()
            self.console.print(Markdown(self.content))


def format_tool_call_preview(tool_name: str, arguments: dict) -> Panel:
    """Format a preview of a tool call before execution."""
    icon = TOOL_ICONS.get(tool_name, "🔧")

    header = Text.assemble(
        (icon, "info"),
        " ",
        (tool_name, "tool_name"),
        " ",
        ("...", "muted"),
    )

    content_parts: list[Text | Syntax] = []

    if arguments:
        json_str = json.dumps(arguments, indent=2, ensure_ascii=False)
        syntax = Syntax(
            json_str,
            "json",
            theme="monokai",
            background_color="default",
        )
        content_parts.append(syntax)

    return Panel(
        Group(*content_parts) if content_parts else Text("No arguments", style="muted"),
        title=header,
        border_style="tool_border dim",
        box=ROUNDED,
        padding=(0, 1),
    )


def format_error(message: str, details: str | None = None) -> Panel:
    """Format an error message."""
    content_parts: list[Text] = [
        Text(f"✗ {message}", style="error bold"),
    ]

    if details:
        content_parts.append(Text(""))
        content_parts.append(Text(details, style="error"))

    return Panel(
        Group(*content_parts),
        border_style="red",
        box=ROUNDED,
        padding=(0, 1),
    )


def format_success(message: str) -> Text:
    """Format a success message."""
    return Text(f"✓ {message}", style="success")


def print_separator(console: Console) -> None:
    """Print a horizontal separator line."""
    console.print(Text("─" * console.width, style="separator"))


def print_welcome_banner(console: Console, version: str = "0.1.0") -> None:
    """Print a welcome banner on startup."""
    from rich.align import Align
    from rich.rule import Rule

    # ASCII art logo - properly aligned (no indentation)
    logo = """╔═╗╔═╗╔═╗╔╗╔╔╗╔╔╦╗
╠═╣╠═╝╠═╣║║║║║║ ║
╩ ╩╩  ╩ ╩╝╚╝╝╚╝ ╩"""

    # Build welcome text
    welcome_text = Text.assemble(
        ("\n" + logo + "\n", "bright_cyan bold"),
        ("\n  AI-powered coding assistant\n", "bright_white"),
        (f"  Version {version}\n", "muted"),
    )

    # Tips section
    tips_table = Table(show_header=False, box=None, padding=(0, 1))
    tips_table.add_column("bullet", style="bright_cyan", justify="center")
    tips_table.add_column("tip", style="bright_white")

    tips = [
        ("•", "Type your task and press Enter"),
        ("•", "Use Alt+Enter for multi-line input"),
        ("•", "Press Ctrl+C to exit at any time"),
        ("•", "Run with --shortcuts to see all keyboard shortcuts"),
    ]

    for bullet, tip in tips:
        tips_table.add_row(bullet, tip)

    # Combine everything
    content = Group(
        Align.center(welcome_text),
        Rule(style="dim"),
        Align.center(tips_table),
        Rule(style="dim"),
    )

    console.print()
    console.print(content)
    console.print()


def print_status_bar(
    console: Console,
    model: str,
    workspace: str,
    provider: str = "",
) -> None:
    """Print a status bar showing current session info."""
    from rich.rule import Rule

    # Truncate long paths
    display_workspace = workspace if len(workspace) < 40 else "..." + workspace[-37:]

    status = Text.assemble(
        ("Model: ", "muted"),
        (model, "bright_cyan"),
        (" │ ", "dim"),
        ("Workspace: ", "muted"),
        (display_workspace, "bright_green"),
    )

    if provider:
        status.append_text(Text.assemble(
            (" │ ", "dim"),
            ("Provider: ", "muted"),
            (provider, "bright_magenta"),
        ))

    console.print(Rule(style="dim"))
    console.print(status)
    console.print(Rule(style="dim"))


def print_goodbye(console: Console) -> None:
    """Print a goodbye message on exit."""
    console.print()
    console.print(
        Text.assemble(
            ("👋 ", "info"),
            ("Goodbye! Session saved.", "bright_white"),
        )
    )
    console.print()


def print_tool_timeline(console: Console, tool_calls: list[dict]) -> None:
    """Print a timeline of tool calls."""
    from rich.tree import Tree

    if not tool_calls:
        return

    tree = Tree("[bright_magenta]Tool Calls[/bright_magenta]")

    for call in tool_calls:
        name = call.get("name", "unknown")
        status = call.get("status", "pending")
        duration = call.get("duration_ms", 0)

        icon = TOOL_ICONS.get(name, "🔧")
        status_icon = "✓" if status == "success" else "❌" if status == "error" else "⏳"

        label = f"{icon} {name} {status_icon}"
        if duration:
            label += f" ({duration:.0f}ms)"

        style = "success" if status == "success" else "error" if status == "error" else "muted"
        tree.add(f"[{style}]{label}[/{style}]")

    console.print(tree)


def print_input_prompt(console: Console, message: str = "You") -> None:
    """Print a styled input prompt."""
    from rich.rule import Rule

    # Print a subtle separator before input
    console.print()
    console.print(
        Text.assemble(
            ("➜ ", "bright_green"),
            (message, "bright_white bold"),
        ),
        end=""
    )


def print_response_prefix(console: Console) -> None:
    """Print a prefix before assistant response."""
    console.print()
    console.print(
        Text.assemble(
            ("◆ ", "bright_magenta"),
            ("Agent", "bright_magenta bold"),
        )
    )


class ConversationDisplay:
    """Display messages in a conversation format like Claude Code."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._message_count = 0

    def print_user_message(self, content: str, show_header: bool = True) -> None:
        """Print a user message."""
        self._message_count += 1
        if show_header:
            self.console.print()
            self.console.print(
                Text.assemble(
                    ("➜ ", "bright_green"),
                    ("You", "bright_green bold"),
                )
            )
        self.console.print(Markdown(content))

    def print_assistant_message(self, content: str, show_header: bool = True) -> None:
        """Print an assistant message with markdown rendering."""
        self._message_count += 1
        if show_header:
            self.console.print()
            self.console.print(
                Text.assemble(
                    ("◆ ", "bright_magenta"),
                    ("Agent", "bright_magenta bold"),
                )
            )
        self.console.print(Markdown(content, code_theme="monokai"))

    def print_system_message(self, content: str, style: str = "info") -> None:
        """Print a system message."""
        self.console.print(
            Text.assemble(
                ("ℹ ", style),
                (content, f"{style} dim"),
            )
        )

    def print_tool_call(self, tool_name: str, arguments: dict, status: str = "running") -> None:
        """Print a tool call notification."""
        icon = TOOL_ICONS.get(tool_name, "🔧")
        if status == "running":
            self.console.print(
                Text.assemble(
                    ("  ", ""),
                    (icon, "info"),
                    (f" {tool_name}", "tool_name dim"),
                    (" ...", "muted"),
                )
            )


class CollapsibleSection:
    """A collapsible section for long outputs (simulated with panels)."""

    def __init__(
        self,
        console: Console,
        title: str,
        content: str,
        collapsed: bool = False,
        max_lines: int = 20,
    ) -> None:
        self.console = console
        self.title = title
        self.content = content
        self.collapsed = collapsed
        self.max_lines = max_lines

    def display(self) -> None:
        """Display the section."""
        lines = self.content.split("\n")
        total_lines = len(lines)

        if self.collapsed or total_lines <= self.max_lines:
            display_content = self.content
            truncation_notice = ""
        else:
            display_content = "\n".join(lines[:self.max_lines])
            truncation_notice = f"\n... ({total_lines - self.max_lines} more lines, use --no-truncate to see all)"

        # Try to detect language for syntax highlighting
        content_renderable = format_code_output(display_content)

        panel_content = (
            Group(content_renderable, Text(truncation_notice, style="muted"))
            if truncation_notice
            else content_renderable
        )
        panel = Panel(
            panel_content,
            title=self.title,
            border_style="dim",
            box=ROUNDED,
            padding=(0, 1),
        )
        self.console.print(panel)


def print_conversation_turn(
    console: Console,
    turn_number: int,
    user_content: str,
    assistant_content: str | None = None,
    token_usage: tuple[int, int] | None = None,
    execution_time: float | None = None,
) -> None:
    """Print a complete conversation turn like Claude Code."""
    # Turn separator with number
    console.print()
    console.print(
        Rule(
            title=f"[dim]Turn {turn_number}[/dim]",
            style="dim",
            align="center",
        )
    )

    # User message
    console.print(
        Text.assemble(
            ("➜ ", "bright_green"),
            ("You", "bright_green bold"),
        )
    )
    console.print(Markdown(user_content))

    # Assistant message (if present)
    if assistant_content:
        console.print()
        console.print(
            Text.assemble(
                ("◆ ", "bright_magenta"),
                ("Agent", "bright_magenta bold"),
            )
        )
        console.print(Markdown(assistant_content, code_theme="monokai"))

    # Footer with metadata
    if token_usage or execution_time:
        footer_parts: list[tuple[str, str]] = []
        if execution_time:
            footer_parts.append((f"⏱ {execution_time:.1f}s", "timestamp"))
        if token_usage:
            input_tok, output_tok = token_usage
            footer_parts.append((f"📊 {input_tok}+{output_tok} tokens", "muted"))

        if footer_parts:
            footer = Text.assemble(*[(f"  {text}", style) for text, style in footer_parts])
            console.print(footer)


def print_tool_summary(
    console: Console,
    tools_executed: list[dict],
    total_time_ms: float,
) -> None:
    """Print a summary of all tools executed in a turn."""
    if not tools_executed:
        return

    # Build summary table
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("Tool", style="tool_name")
    table.add_column("Status", style="success")
    table.add_column("Time", justify="right", style="timestamp")

    for tool in tools_executed:
        name = tool.get("name", "unknown")
        status = "✓" if tool.get("success", True) else "✗"
        status_style = "success" if tool.get("success", True) else "error"
        time_ms = tool.get("duration_ms", 0)

        table.add_row(
            f"{TOOL_ICONS.get(name, '🔧')} {name}",
            f"[{status_style}]{status}[/{status_style}]",
            f"{time_ms:.0f}ms" if time_ms else "-",
        )

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bright_magenta]Tools ({len(tools_executed)}) • {total_time_ms:.0f}ms[/bright_magenta]",
            border_style="bright_magenta",
            box=ROUNDED,
            padding=(0, 1),
        )
    )


def print_error_panel(
    console: Console,
    title: str,
    message: str,
    suggestion: str | None = None,
) -> None:
    """Print a prominent error panel."""
    content_parts: list[RenderableType] = [
        Text(message, style="red"),
    ]

    if suggestion:
        content_parts.append(Text(""))
        content_parts.append(Text(f"💡 {suggestion}", style="yellow"))

    console.print(
        Panel(
            Group(*content_parts),
            title=f"[bold red]✗ {title}[/bold red]",
            border_style="red",
            box=ROUNDED,
            padding=(0, 1),
        )
    )


def print_success_panel(
    console: Console,
    title: str,
    message: str | None = None,
) -> None:
    """Print a success panel."""
    content = Text(message, style="green") if message else None

    console.print(
        Panel(
            content or "",
            title=f"[bold green]✓ {title}[/bold green]",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1),
        )
    )


def format_file_tree(
    console: Console,
    paths: list[str],
    root: str = ".",
) -> Panel:
    """Format a file tree display like Claude Code.

    Args:
        console: The rich console
        paths: List of file paths to display
        root: Root directory name
    """
    from rich.tree import Tree

    if not paths:
        return Panel(
            Text("No files", style="muted"),
            title="[bright_cyan]Files[/bright_cyan]",
            border_style="bright_cyan",
            box=ROUNDED,
        )

    # Build tree structure
    tree = Tree(f"[bright_green]{root}[/bright_green]")
    nodes: dict[str, Tree] = {root: tree}

    for path in sorted(paths):
        parts = path.split("/")
        current_path = root
        parent = tree

        for i, part in enumerate(parts):
            current_path = f"{current_path}/{part}" if current_path != root else part

            if current_path not in nodes:
                # Check if it's a file (last part) or directory
                is_file = i == len(parts) - 1
                if is_file:
                    # File icon based on extension
                    icon = "📄"
                    if part.endswith(".py"):
                        icon = "🐍"
                    elif part.endswith(".md"):
                        icon = "📝"
                    elif part.endswith(".json"):
                        icon = "📋"
                    elif part.endswith(".js") or part.endswith(".ts"):
                        icon = "📜"
                    elif part.endswith(".html"):
                        icon = "🌐"
                    elif part.endswith(".css"):
                        icon = "🎨"
                    node = parent.add(f"{icon} [bright_white]{part}[/bright_white]")
                else:
                    node = parent.add(f"📁 [bright_cyan]{part}[/bright_cyan]")
                nodes[current_path] = node

            parent = nodes[current_path]

    return Panel(
        tree,
        title=f"[bright_cyan]Files ({len(paths)})[/bright_cyan]",
        border_style="bright_cyan",
        box=ROUNDED,
        padding=(0, 1),
    )


def format_diff(
    console: Console,
    old_content: str,
    new_content: str,
    filepath: str = "",
) -> Panel:
    """Format a diff display like Claude Code.

    Args:
        console: The rich console
        old_content: Original content
        new_content: Modified content
        filepath: Optional file path for the title
    """
    import difflib

    # Generate unified diff
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    if not old_lines[-1].endswith("\n"):
        old_lines[-1] += "\n"
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] += "\n"

    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filepath}" if filepath else "a/file",
        tofile=f"b/{filepath}" if filepath else "b/file",
    ))

    if not diff:
        return Panel(
            Text("No changes", style="muted"),
            title="[bright_cyan]Diff[/bright_cyan]",
            border_style="bright_cyan",
            box=ROUNDED,
        )

    # Format diff with syntax highlighting
    diff_text = "".join(diff)

    return Panel(
        Syntax(
            diff_text,
            "diff",
            theme="monokai",
            background_color="default",
            line_numbers=False,
        ),
        title=f"[bright_cyan]Changes in {filepath}[/bright_cyan]" if filepath else "[bright_cyan]Changes[/bright_cyan]",
        border_style="bright_cyan",
        box=ROUNDED,
        padding=(0, 1),
    )


class ProgressTracker:
    """Track and display progress for multi-step operations."""

    def __init__(self, console: Console, total_steps: int, description: str = "Processing") -> None:
        self.console = console
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self._live: Live | None = None

    def __enter__(self) -> ProgressTracker:
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def _render(self) -> RenderableType:
        """Render the progress display."""
        progress = self.current_step / self.total_steps
        filled = int(progress * 20)
        empty = 20 - filled
        bar = "█" * filled + "░" * empty

        return Text.assemble(
            (f"{self.description} ", "bright_white"),
            (f"[{bar}]", "bright_cyan"),
            (f" {self.current_step}/{self.total_steps}", "muted"),
        )

    def update(self, step: int | None = None, message: str | None = None) -> None:
        """Update progress."""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if message:
            self.description = message

        if self._live:
            self._live.update(self._render())

    def complete(self, message: str = "Complete") -> None:
        """Mark as complete."""
        self.current_step = self.total_steps
        if self._live:
            self._live.update(
                Text.assemble(
                    (f"✓ {message}", "success"),
                )
            )


__all__ = [
    "APPROVAL_STYLES",
    "CLI_THEME",
    "create_console",
    "format_tool_name",
    "format_approval_request",
    "format_question_interrupt",
    "format_tool_output",
    "format_code_output",
    "format_assistant_message",
    "format_system_message",
    "tool_execution_spinner",
    "StreamingMarkdown",
    "StreamingResponse",
    "format_tool_call_preview",
    "format_error",
    "format_success",
    "print_separator",
    "print_welcome_banner",
    "print_status_bar",
    "print_goodbye",
    "print_tool_timeline",
    "print_input_prompt",
    "print_response_prefix",
    "TOOL_ICONS",
    "TOOL_CATEGORIES",
    "ThinkingIndicator",
    "ToolExecutionDisplay",
    "ConversationDisplay",
    "CollapsibleSection",
    "print_conversation_turn",
    "print_tool_summary",
    "print_error_panel",
    "print_success_panel",
    "format_file_tree",
    "format_diff",
    "ProgressTracker",
]

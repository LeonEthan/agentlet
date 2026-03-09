"""Terminal CLI entrypoint for the agentlet runtime app."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Protocol, Sequence, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PromptStyle
from rich.box import ROUNDED
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agentlet.config import apply_env_from_settings, load_settings
from agentlet.core.interrupts import (
    ApprovalRequest,
    ApprovalResponse,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.core.loop import CompletedTurn, InterruptedTurn
from agentlet.runtime.app import (
    RuntimeApp,
    VALID_PROVIDER_NAMES,
    build_default_runtime_app,
)
from agentlet.runtime.events import RuntimeEvent

from .history import InputHistory
from .rich_ui import (
    create_console,
    format_approval_request,
    format_error,
    format_question_interrupt,
    format_system_message,
    format_tool_output,
    format_code_output,
    ThinkingIndicator,
    ToolExecutionDisplay,
    TOOL_ICONS,
    print_welcome_banner,
    print_status_bar,
    print_goodbye,
    StreamingResponse,
    print_input_prompt,
    print_response_prefix,
    ConversationDisplay,
    print_conversation_turn,
    print_tool_summary,
    print_error_panel,
)


CLI_PROVIDER_CHOICES = (
    "anthropic",
    "openai",
    "openai-like",
    *sorted(VALID_PROVIDER_NAMES),
)

# Prompt toolkit style matching our Rich theme
PROMPT_STYLE = PromptStyle.from_dict({
    "prompt": "ansicyan bold",
    "": "ansiwhite",
})


def is_interactive(stdin: TextIO | None = None) -> bool:
    """Check if stdin is a TTY (interactive terminal)."""
    return (stdin or sys.stdin).isatty()


class RuntimeAppFactory(Protocol):
    """Factory contract used by the CLI to assemble the runtime app."""

    def __call__(self, args: argparse.Namespace, user_io: "TerminalUserIO") -> RuntimeApp:
        """Build one configured runtime app."""


class TerminalUserIO:
    """Terminal adapter implementing the runtime user interaction contract.

    Provides a Claude Code-like rich terminal experience with:
    - Styled tool execution display
    - Real-time streaming responses
    - Rich formatting for approvals and questions
    - Enhanced error display
    - Graceful fallback for non-interactive mode
    """

    def __init__(
        self,
        *,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO,
        console: Console | None = None,
        auto_approve: bool = False,
        interactive: bool | None = None,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        # Create console that writes to the provided stdout for test compatibility
        self.console = console or create_console(file=stdout)
        self.events: list[RuntimeEvent] = []
        # Allow forcing interactive mode (for tests) or auto-detect
        self._interactive = interactive if interactive is not None else is_interactive(stdin)
        self._auto_approve = auto_approve
        self._prompt_session: PromptSession[str] | None = None

    def _get_prompt_session(self) -> PromptSession[str] | None:
        """Get or create the prompt session with history (interactive only)."""
        if not self._interactive:
            return None
        if self._prompt_session is None:
            self._prompt_session = PromptSession(style=PROMPT_STYLE)
        return self._prompt_session

    def emit_event(self, event: RuntimeEvent) -> None:
        """Emit a runtime event."""
        self.events.append(event)

        if event.kind == "resumed":
            self.console.print(format_system_message("Resuming...", style="info"))

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        """Display an approval request and collect user decision."""
        # In non-interactive mode with auto-approve, automatically approve read-only
        if not self._interactive and self._auto_approve:
            if request.approval_category == "read_only":
                self.console.print(format_system_message(
                    f"Auto-approved {request.tool_name} (read-only)"
                ))
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="approved",
                )

        # Render the approval panel
        panel = format_approval_request(request)
        self.console.print(panel)

        # In non-interactive mode without auto-approve, reject by default
        if not self._interactive:
            self.console.print(Text(
                "⚠ Non-interactive mode: approval required but no TTY available. "
                "Use --auto-approve to approve read-only operations automatically.",
                style="warning"
            ))
            return ApprovalResponse(
                request_id=request.request_id,
                decision="rejected",
            )

        # Interactive mode: collect user input
        # For tests with StringIO, use self.stdin.readline() instead of input()
        use_simple_input = not hasattr(self.stdin, 'isatty') or not self.stdin.isatty()

        while True:
            try:
                if use_simple_input:
                    decision = self.stdin.readline()
                else:
                    session = self._get_prompt_session()
                    if session:
                        decision = session.prompt("")
                    else:
                        decision = input()
            except (EOFError, KeyboardInterrupt):
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="rejected",
                )

            normalized = decision.strip().lower()

            if normalized in {"y", "yes"}:
                self.console.print(Text("✓ Approved", style="success"))
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="approved",
                )
            if normalized in {"", "n", "no"}:
                self.console.print(Text("✗ Rejected", style="rejected"))
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="rejected",
                )

            self.console.print(
                Text("Please answer yes or no.", style="warning")
            )

    def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
        """Surface a structured question."""
        panel = format_question_interrupt(request)
        self.console.print(panel)

    def resolve_question_interrupt(
        self,
        request: UserQuestionRequest,
    ) -> UserQuestionResponse:
        """Collect a structured answer for a question interrupt."""
        if not request.options and not request.allow_free_text:
            raise RuntimeError(
                "Question interrupt has no options and does not allow free text."
            )

        # Non-interactive mode: try to use first option or fail
        if not self._interactive:
            if request.options:
                default = request.options[0].value
                self.console.print(Text(
                    f"Non-interactive mode: auto-selecting first option '{default}'",
                    style="warning"
                ))
                return UserQuestionResponse(
                    request_id=request.request_id,
                    selected_option=default,
                )
            raise RuntimeError(
                "Question interrupt in non-interactive mode with no options to auto-select"
            )

        # Interactive mode
        # Use simple input() for tests with StringIO (no real isatty)
        use_simple_input = not hasattr(self.stdin, 'isatty') or not self.stdin.isatty()

        while True:
            try:
                if use_simple_input:
                    self.stdout.write("Answer: ")
                    self.stdout.flush()
                    answer = self.stdin.readline()
                    # Check for EOF (empty string returned by readline at EOF)
                    if answer == "":
                        raise RuntimeError(
                            "Input closed while waiting for a question response."
                        )
                else:
                    session = self._get_prompt_session()
                    if session:
                        answer = session.prompt(
                            Text.assemble(("➜ ", "prompt")).plain,
                        )
                    else:
                        answer = input(Text.assemble(("➜ ", "prompt")).plain + " ")
            except (EOFError, KeyboardInterrupt) as exc:
                raise RuntimeError(
                    "Input closed while waiting for a question response."
                ) from exc

            answer = answer.strip()

            # Check for indexed option selection
            if answer.isdigit():
                index = int(answer)
                if 1 <= index <= len(request.options):
                    selected = request.options[index - 1].value
                    self.console.print(
                        Text.assemble(
                            ("✓ Selected: ", "success"),
                            (request.options[index - 1].label, "bright_cyan"),
                        )
                    )
                    return UserQuestionResponse(
                        request_id=request.request_id,
                        selected_option=selected,
                    )

            # Check for direct value match
            for option in request.options:
                if option.value == answer:
                    self.console.print(
                        Text.assemble(
                            ("✓ Selected: ", "success"),
                            (option.label, "bright_cyan"),
                        )
                    )
                    return UserQuestionResponse(
                        request_id=request.request_id,
                        selected_option=answer,
                    )

            # Check for free text
            if request.allow_free_text and answer:
                return UserQuestionResponse(
                    request_id=request.request_id,
                    free_text=answer,
                )

            self.console.print(
                Text("Please choose a listed option or enter free text.", style="warning")
            )

    def print_assistant_message(self, content: str) -> None:
        """Print a completed assistant message with markdown rendering."""
        self.console.print(Markdown(content))

    def show_thinking(self, message: str = "Thinking..."):
        """Show a thinking indicator. Returns a context manager."""
        return ThinkingIndicator(self.console, message)

    def print_error(self, message: str, details: str | None = None) -> None:
        """Print an error message."""
        self.console.print(format_error(message, details))

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(format_system_message(message))

    def display_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Display a tool call before execution."""
        from rich.syntax import Syntax
        from rich.panel import Panel
        from rich.text import Text

        icon = TOOL_ICONS.get(tool_name, "🔧")
        header = Text.assemble(
            (icon, "info"),
            " ",
            (tool_name, "tool_name"),
            " ",
            ("...", "muted"),
        )

        if arguments:
            import json
            json_str = json.dumps(arguments, indent=2, ensure_ascii=False)
            content = Syntax(json_str, "json", theme="monokai", background_color="default")
        else:
            content = Text("No arguments", style="muted")

        self.console.print(Panel(
            content,
            title=header,
            border_style="tool_border dim",
            box=ROUNDED,
            padding=(0, 1),
        ))

    def display_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool = False,
        execution_time_ms: float | None = None,
    ) -> None:
        """Display a tool execution result."""
        panel = format_tool_output(
            tool_name, result, is_error=is_error, execution_time_ms=execution_time_ms
        )
        self.console.print(panel)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    app_factory: RuntimeAppFactory | None = None,
) -> int:
    """Run the terminal app with runtime-owned orchestration and approvals."""
    console = create_console()

    # 1. Load user settings and apply environment variables
    try:
        settings = load_settings()
        apply_env_from_settings(settings)
    except ValueError as exc:
        console.print(format_error("Configuration error", str(exc)))
        return 2

    # 2. Build parser with settings defaults
    parser = _build_parser(defaults=settings.defaults)
    args = parser.parse_args(list(argv) if argv is not None else None)

    resolved_stdin = stdin or sys.stdin
    resolved_stdout = stdout or sys.stdout
    resolved_stderr = stderr or sys.stderr

    # Handle --shortcuts flag
    if args.shortcuts:
        shortcuts_console = create_console(file=resolved_stdout)
        print_keyboard_shortcuts_help(shortcuts_console)
        return 0

    # Check if we're in interactive mode
    interactive = is_interactive(resolved_stdin)

    # Create console that writes to the provided stdout for test compatibility
    console = create_console(file=resolved_stdout)

    # Show welcome banner for interactive mode when no direct task
    if interactive and args.task is None:
        print_welcome_banner(console)

    if not interactive and not args.auto_approve:
        console.print(format_system_message(
            "Running in non-interactive mode. Use --auto-approve to approve read-only operations."
        ))

    user_io = TerminalUserIO(
        stdin=resolved_stdin,
        stdout=resolved_stdout,
        stderr=resolved_stderr,
        console=console,
        auto_approve=args.auto_approve,
    )

    # 3. Get task input
    task = _resolve_task(
        args.task,
        user_io=user_io,
        interactive=interactive,
    )
    if not task.strip():
        console.print(format_error("Task must not be empty"))
        return 2

    # 4. Run the runtime
    try:
        # Check if we should run in REPL mode
        if args.repl or (interactive and args.task is None):
            return run_repl_mode(args, user_io, console, app_factory)

        # Single-turn mode
        runtime_app = (
            app_factory(args, user_io)
            if app_factory is not None
            else _build_runtime_app_from_args(args, user_io)
        )

        # Show status bar with session info
        model_name = getattr(settings, 'env', {}).get('AGENTLET_MODEL', 'default')
        provider_name = args.provider or getattr(settings, 'defaults', {}).get('provider', 'openai-like')
        print_status_bar(console, model_name, args.workspace_root, provider_name)

        # Use conversation display for consistent styling
        conversation = ConversationDisplay(console)

        # Show user message
        console.print()
        conversation.print_user_message(task, show_header=True)

        # Run with thinking indicator
        with ThinkingIndicator(console, "Thinking..."):
            outcome = runtime_app.run_turn(current_task=task)

        # Render outcome
        if isinstance(outcome, CompletedTurn):
            if outcome.message.content:
                conversation.print_assistant_message(outcome.message.content, show_header=True)

            # Show success footer
            console.print()
            console.print(
                Rule(style="dim")
            )

            # Show goodbye message on successful completion in interactive mode
            if interactive:
                print_goodbye(console)

        elif isinstance(outcome, InterruptedTurn):
            if outcome.assistant_message and outcome.assistant_message.content:
                conversation.print_assistant_message(outcome.assistant_message.content, show_header=True)

            console.print()
            console.print(
                Panel(
                    Text("Execution paused awaiting more user input.", style="warning"),
                    border_style="warning",
                    box=ROUNDED,
                )
            )

    except ValueError as exc:
        console.print()
        console.print(
            Text.assemble(
                ("◆ ", "bright_magenta"),
                ("Agent", "bright_magenta bold"),
            )
        )
        print_error_panel(console, "Configuration Error", str(exc))
        return 2
    except RuntimeError as exc:
        console.print()
        console.print(
            Text.assemble(
                ("◆ ", "bright_magenta"),
                ("Agent", "bright_magenta bold"),
            )
        )
        print_error_panel(console, "Runtime Error", str(exc))
        return 1
    except Exception as exc:
        console.print()
        console.print(
            Text.assemble(
                ("◆ ", "bright_magenta"),
                ("Agent", "bright_magenta bold"),
            )
        )
        print_error_panel(console, "Unexpected Error", str(exc))
        return 1

    return 0


def print_keyboard_shortcuts_help(console: Console | None = None) -> None:
    """Print keyboard shortcuts help panel."""
    from rich.table import Table
    from rich.align import Align

    console = console or create_console()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("shortcut", style="bright_cyan bold", justify="right")
    table.add_column("description", style="bright_white")

    shortcuts = [
        ("Alt+Enter", "Submit multi-line input"),
        ("Esc then Enter", "Alternative submit"),
        ("Ctrl+C", "Cancel / Exit"),
        ("Ctrl+D", "EOF / Exit"),
        ("↑ / ↓", "Navigate input history"),
        ("Ctrl+R", "Search history (when using prompt)"),
    ]

    for shortcut, desc in shortcuts:
        table.add_row(shortcut, desc)

    panel = Panel(
        table,
        title="[bold bright_blue]Keyboard Shortcuts[/bold bright_blue]",
        border_style="bright_blue",
        box=ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def _handle_slash_command(
    command: str,
    console: Console,
    history: InputHistory,
) -> bool:
    """Handle slash commands. Returns True if should continue, False to exit."""
    cmd = command.strip().lower()

    if cmd == "/exit" or cmd == "/quit":
        print_goodbye(console)
        return False

    if cmd == "/clear":
        console.clear()
        return True

    if cmd == "/history":
        entries = history.get_entries()
        if entries:
            console.print("\n[bold bright_cyan]Input History:[/bold bright_cyan]")
            for i, entry in enumerate(entries[-20:], 1):  # Show last 20
                console.print(f"  {i}. {entry[:80]}{'...' if len(entry) > 80 else ''}")
            console.print()
        else:
            console.print("\n[dim]No history available.[/dim]\n")
        return True

    if cmd == "/help":
        print_slash_commands_help(console)
        return True

    if cmd == "/model":
        # Will be shown in status bar, just acknowledge
        console.print("\n[dim]Model info shown in status bar above.[/dim]\n")
        return True

    # Unknown command
    console.print(f"\n[red]Unknown command: {command}[/red]")
    console.print("[dim]Type /help for available commands.[/dim]\n")
    return True


def print_slash_commands_help(console: Console | None = None) -> None:
    """Print slash commands help."""
    console = console or create_console()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("command", style="bright_cyan bold")
    table.add_column("description", style="bright_white")

    commands = [
        ("/help", "Show this help message"),
        ("/exit", "Exit the REPL"),
        ("/clear", "Clear the screen"),
        ("/history", "Show recent input history"),
        ("/model", "Show current model info"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    panel = Panel(
        table,
        title="[bold bright_blue]Slash Commands[/bold bright_blue]",
        border_style="bright_blue",
        box=ROUNDED,
        padding=(0, 1),
    )
    console.print(panel)


def run_repl_mode(
    args: argparse.Namespace,
    user_io: TerminalUserIO,
    console: Console,
    app_factory: RuntimeAppFactory | None = None,
) -> int:
    """Run the agent in REPL mode for multi-turn conversations."""
    from rich.rule import Rule
    import time

    history = InputHistory()
    session = PromptSession(
        history=FileHistory(str(Path.home() / ".agentlet" / "history_repl")),
        style=PROMPT_STYLE,
        multiline=True,
        wrap_lines=True,
    )

    # Build runtime app once
    try:
        runtime_app = (
            app_factory(args, user_io)
            if app_factory is not None
            else _build_runtime_app_from_args(args, user_io)
        )
    except Exception as exc:
        print_error_panel(console, "Failed to initialize", str(exc))
        return 1

    # Show welcome and help hint
    print_welcome_banner(console)
    console.print("[dim]Type /help for available commands, /exit to quit.[/dim]\n")

    # Show status bar
    model_name = getattr(args, 'model', 'default') or 'default'
    provider_name = args.provider or "default"
    print_status_bar(console, model_name, args.workspace_root, provider_name)

    turn_count = 0
    conversation = ConversationDisplay(console)

    while True:
        turn_count += 1
        turn_start_time = time.time()

        try:
            # Get input with styled prompt (Claude Code style)
            user_input = session.prompt(
                Text.assemble(
                    ("\n", ""),
                    ("➜ ", "bright_green"),
                    ("You", "bright_green bold"),
                    (" ", ""),
                ),
                rprompt="",
            )

            user_input = user_input.strip()

            if not user_input:
                continue

            # Handle slash commands
            if user_input.startswith("/"):
                if not _handle_slash_command(user_input, console, history):
                    return 0
                continue

            # Add to history
            history.add(user_input)

            # Show user message (already shown by prompt, but add spacing)
            console.print()  # Space before agent response

            # Show thinking indicator and run
            assistant_content = None
            with ThinkingIndicator(console, "Thinking..."):
                outcome = runtime_app.run_turn(current_task=user_input)

            turn_duration = time.time() - turn_start_time

            # Render outcome with conversation style
            if isinstance(outcome, CompletedTurn):
                assistant_content = outcome.message.content
                if assistant_content:
                    conversation.print_assistant_message(assistant_content, show_header=True)

                # Show turn footer with timing
                console.print()
                console.print(
                    Rule(
                        style="dim",
                    )
                )

            elif isinstance(outcome, InterruptedTurn):
                assistant_content = outcome.assistant_message.content if outcome.assistant_message else None
                if assistant_content:
                    conversation.print_assistant_message(assistant_content, show_header=True)

                console.print()
                console.print(
                    Panel(
                        Text("Execution paused awaiting more user input.", style="warning"),
                        border_style="warning",
                        box=ROUNDED,
                    )
                )

        except KeyboardInterrupt:
            console.print("\n[dim](Interrupted - type /exit to quit)[/dim]\n")
            continue
        except EOFError:
            print_goodbye(console)
            return 0
        except Exception as exc:
            console.print()
            console.print(
                Text.assemble(
                    ("◆ ", "bright_magenta"),
                    ("Agent", "bright_magenta bold"),
                )
            )
            print_error_panel(console, "Error", str(exc))
            continue


def _build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build argument parser with optional defaults from settings file."""
    defaults = defaults or {}

    parser = argparse.ArgumentParser(
        prog="agentlet",
        description="AI-powered coding assistant with rich terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agentlet "Explain this codebase"
  agentlet --workspace-root /path/to/project "Find bugs"
  agentlet --provider anthropic "Refactor main.py"
  agentlet --auto-approve "Run tests non-interactively"
        """.strip(),
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to hand to the agent. If not provided, you'll be prompted.",
    )
    parser.add_argument(
        "--workspace-root",
        default=defaults.get("workspace_root", "."),
        help="Workspace directory exposed to the built-in coding tools. (default: .)",
    )
    parser.add_argument(
        "--state-dir",
        default=defaults.get("state_dir", ".agentlet"),
        help="Directory used for session and memory files. (default: .agentlet)",
    )
    parser.add_argument(
        "--session-path",
        default=defaults.get("session_path"),
        help="JSONL session history path. Defaults under the state directory.",
    )
    parser.add_argument(
        "--memory-path",
        default=defaults.get("memory_path"),
        help="Markdown durable memory path. Defaults under the state directory.",
    )
    parser.add_argument(
        "--instructions-path",
        default=defaults.get("instructions_path"),
        help="Instructions file path. Defaults to AGENTS.md in the workspace.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=defaults.get("max_iterations"),
        help="Maximum tool call iterations per turn. (default: 8)",
    )
    parser.add_argument(
        "--bash-timeout-seconds",
        type=float,
        default=defaults.get("bash_timeout_seconds"),
        help="Default timeout for Bash tool execution.",
    )
    parser.add_argument(
        "--provider",
        choices=tuple(dict.fromkeys(CLI_PROVIDER_CHOICES)),
        default=defaults.get("provider"),
        help="LLM provider to use.",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically approve read-only operations (useful for non-interactive mode).",
    )
    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich formatting and use plain text output.",
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start in REPL mode for interactive multi-turn conversations.",
    )
    parser.add_argument(
        "--shortcuts",
        action="store_true",
        help="Show keyboard shortcuts help and exit.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
        help="Show version information and exit.",
    )
    return parser


def _build_runtime_app_from_args(
    args: argparse.Namespace,
    user_io: TerminalUserIO,
) -> RuntimeApp:
    """Build the runtime app from parsed CLI arguments."""
    kwargs: dict[str, Any] = {}
    if args.max_iterations is not None:
        kwargs["max_iterations"] = args.max_iterations
    if args.bash_timeout_seconds is not None:
        kwargs["bash_timeout_seconds"] = args.bash_timeout_seconds
    return build_default_runtime_app(
        user_io=user_io,
        workspace_root=args.workspace_root,
        state_dir=args.state_dir,
        session_path=args.session_path,
        memory_path=args.memory_path,
        instructions_path=args.instructions_path,
        provider=args.provider,
        **kwargs,
    )


def _resolve_task(
    task: str | None,
    *,
    user_io: TerminalUserIO,
    interactive: bool,
) -> str:
    """Resolve the task from arguments or interactive prompt."""
    if task is not None:
        return task

    if not interactive:
        # Try to read from stdin in non-interactive mode
        stdin = user_io.stdin
        try:
            import select
            # Check if stdin has data available (needs real fileno, fails in some test setups)
            if hasattr(stdin, 'fileno') and select.select([stdin], [], [], 0.0)[0]:
                return stdin.read().strip()
        except (OSError, ValueError, TypeError):
            # stdin doesn't support select (e.g., StringIO in tests)
            pass
        # Fall back to reading directly if there's data
        content = stdin.read().strip()
        return content

    # Use prompt toolkit for interactive input with persistent history
    history = InputHistory()
    session = PromptSession(
        history=FileHistory(str(Path.home() / ".agentlet" / "history_prompt")),
        style=PROMPT_STYLE,
        multiline=True,
        wrap_lines=True,
    )

    user_io.console.print(
        Panel(
            Text.assemble(
                ("💡 ", "info"),
                ("Enter your task below.", "bright_white"),
                "\n",
                ("Press ", "muted"),
                ("Alt+Enter", "prompt"),
                (" or ", "muted"),
                ("Esc then Enter", "prompt"),
                (" to submit.", "muted"),
            ),
            border_style="info",
            box=ROUNDED,
        )
    )

    try:
        result = session.prompt(
            Text.assemble(("Task: ", "prompt")).plain,
            rprompt="",
        )
        # Save to persistent history
        history.add(result)
        return result
    except (EOFError, KeyboardInterrupt):
        return ""


def _render_outcome(
    outcome: CompletedTurn | InterruptedTurn,
    *,
    user_io: TerminalUserIO,
) -> int:
    """Render the outcome of a turn."""
    if isinstance(outcome, CompletedTurn):
        if outcome.message.content:
            user_io.console.print()
            user_io.print_assistant_message(outcome.message.content)
        return 0

    # Interrupted turn
    user_io.console.print()
    user_io.console.print(
        Panel(
            Text("Execution paused awaiting more user input.", style="warning"),
            border_style="warning",
            box=ROUNDED,
        )
    )
    return 0


__all__ = [
    "main",
    "TerminalUserIO",
    "run_repl_mode",
    "print_keyboard_shortcuts_help",
    "is_interactive",
]

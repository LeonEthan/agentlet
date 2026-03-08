"""Terminal CLI entrypoint for the agentlet runtime app."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Protocol, Sequence, TextIO

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


CLI_PROVIDER_CHOICES = (
    "anthropic",
    "openai",
    "openai-like",
    *sorted(VALID_PROVIDER_NAMES),
)


class RuntimeAppFactory(Protocol):
    """Factory contract used by the CLI to assemble the runtime app."""

    def __call__(self, args: argparse.Namespace, user_io: "TerminalUserIO") -> RuntimeApp:
        """Build one configured runtime app."""


class TerminalUserIO:
    """Terminal adapter implementing the runtime user interaction contract."""

    def __init__(
        self,
        *,
        stdin: TextIO,
        stdout: TextIO,
        stderr: TextIO,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.events: list[RuntimeEvent] = []

    def emit_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def request_approval(self, request: ApprovalRequest) -> ApprovalResponse:
        self.stdout.write(f"Approval required for {request.tool_name}\n")
        self.stdout.write(f"{request.prompt}\n")
        if request.arguments:
            rendered_arguments = json.dumps(
                request.arguments,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            self.stdout.write(f"Arguments:\n{rendered_arguments}\n")
        self.stdout.flush()

        while True:
            self.stdout.write("Approve? [y/N]: ")
            self.stdout.flush()
            decision = self.stdin.readline()
            if decision == "":
                normalized = ""
            else:
                normalized = decision.strip().lower()

            if normalized in {"y", "yes"}:
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="approved",
                )
            if normalized in {"", "n", "no"}:
                return ApprovalResponse(
                    request_id=request.request_id,
                    decision="rejected",
                )

            self.stdout.write("Please answer yes or no.\n")
            self.stdout.flush()

    def begin_question_interrupt(self, request: UserQuestionRequest) -> None:
        self.stdout.write("Agent needs clarification before continuing.\n")
        self.stdout.write(f"{request.prompt}\n")
        for index, option in enumerate(request.options, start=1):
            self.stdout.write(f"{index}. {option.label} [{option.value}]\n")
        if request.allow_free_text:
            self.stdout.write("Free-text answers are allowed.\n")
        self.stdout.flush()

    def resolve_question_interrupt(
        self,
        request: UserQuestionRequest,
    ) -> UserQuestionResponse:
        if not request.options and not request.allow_free_text:
            raise RuntimeError(
                "Question interrupt has no options and does not allow free text."
            )
        while True:
            self.stdout.write("Answer: ")
            self.stdout.flush()
            raw_answer = self.stdin.readline()
            if raw_answer == "":
                raise RuntimeError(
                    "Input closed while waiting for a question response."
                )
            answer = raw_answer.strip()

            indexed_option = _resolve_indexed_option(request, answer)
            if indexed_option is not None:
                return UserQuestionResponse(
                    request_id=request.request_id,
                    selected_option=indexed_option,
                )

            if any(option.value == answer for option in request.options):
                return UserQuestionResponse(
                    request_id=request.request_id,
                    selected_option=answer,
                )

            if request.allow_free_text and answer:
                return UserQuestionResponse(
                    request_id=request.request_id,
                    free_text=answer,
                )

            self.stdout.write("Please choose a listed option or enter free text.\n")
            self.stdout.flush()


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    app_factory: RuntimeAppFactory | None = None,
) -> int:
    """Run the terminal app with runtime-owned orchestration and approvals."""

    # 1. Load user settings and apply environment variables
    try:
        settings = load_settings()
        apply_env_from_settings(settings)
    except ValueError as exc:
        sys.stderr.write(f"Configuration error: {exc}\n")
        sys.stderr.flush()
        return 2

    # 2. Build parser with settings defaults
    parser = _build_parser(defaults=settings.defaults)
    args = parser.parse_args(list(argv) if argv is not None else None)
    resolved_stdin = stdin or sys.stdin
    resolved_stdout = stdout or sys.stdout
    resolved_stderr = stderr or sys.stderr
    user_io = TerminalUserIO(
        stdin=resolved_stdin,
        stdout=resolved_stdout,
        stderr=resolved_stderr,
    )

    task = _resolve_task(args.task, stdin=resolved_stdin, stdout=resolved_stdout)
    if not task.strip():
        resolved_stderr.write("Task must not be empty.\n")
        resolved_stderr.flush()
        return 2

    try:
        runtime_app = (
            app_factory(args, user_io)
            if app_factory is not None
            else _build_runtime_app_from_args(args, user_io)
        )
        outcome = runtime_app.run_turn(current_task=task)
    except ValueError as exc:
        resolved_stderr.write(f"{exc}\n")
        resolved_stderr.flush()
        return 2
    except RuntimeError as exc:
        resolved_stderr.write(f"{exc}\n")
        resolved_stderr.flush()
        return 1

    return _render_outcome(outcome, stdout=resolved_stdout)


def _build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build argument parser with optional defaults from settings file."""
    defaults = defaults or {}

    parser = argparse.ArgumentParser(prog="agentlet")
    parser.add_argument("task", nargs="?", help="Task to hand to the agent.")
    parser.add_argument(
        "--workspace-root",
        default=defaults.get("workspace_root", "."),
        help="Workspace directory exposed to the built-in coding tools.",
    )
    parser.add_argument(
        "--state-dir",
        default=defaults.get("state_dir", ".agentlet"),
        help="Directory used for session and memory files when explicit paths are omitted.",
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
        help="Instructions file path. Defaults to AGENTS.md in the workspace when present.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=defaults.get("max_iterations"),
        help="Maximum tool call iterations per turn.",
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
    return parser


def _build_runtime_app_from_args(
    args: argparse.Namespace,
    user_io: TerminalUserIO,
) -> RuntimeApp:
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
    stdin: TextIO,
    stdout: TextIO,
) -> str:
    if task is not None:
        return task
    stdout.write("Task: ")
    stdout.flush()
    return stdin.readline().rstrip("\n")


def _render_outcome(
    outcome: CompletedTurn | InterruptedTurn,
    *,
    stdout: TextIO,
) -> int:
    if isinstance(outcome, CompletedTurn):
        if outcome.message.content:
            stdout.write(f"{outcome.message.content}\n")
            stdout.flush()
        return 0

    stdout.write("Execution paused awaiting more user input.\n")
    stdout.flush()
    return 0


def _resolve_indexed_option(
    request: UserQuestionRequest,
    answer: str,
) -> str | None:
    if not answer.isdigit():
        return None
    index = int(answer)
    if index < 1 or index > len(request.options):
        return None
    return request.options[index - 1].value


__all__ = ["TerminalUserIO", "main"]

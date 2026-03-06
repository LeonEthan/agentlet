"""Built-in Bash execution tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult

_BASH_EXECUTABLE = "/bin/bash"


@dataclass(frozen=True, slots=True)
class BashExecution:
    """Structured subprocess result used by the Bash tool."""

    command: str
    cwd: str
    timeout_seconds: float | None
    started_at: float
    duration_seconds: float
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False

    def as_metadata(self) -> JSONObject:
        return {
            "command": self.command,
            "cwd": self.cwd,
            "timeout_seconds": self.timeout_seconds,
            "started_at": self.started_at,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
        }


def run_bash_command(
    command: str,
    *,
    cwd: Path,
    timeout_seconds: float | None = None,
) -> BashExecution:
    """Run a command through bash with explicit cwd and timeout semantics."""

    started_at = time.time()
    started_monotonic = time.monotonic()

    try:
        completed = subprocess.run(
            [_BASH_EXECUTABLE, "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return BashExecution(
            command=command,
            cwd=str(cwd),
            timeout_seconds=timeout_seconds,
            started_at=started_at,
            duration_seconds=time.monotonic() - started_monotonic,
            stdout=_coerce_process_output(exc.stdout),
            stderr=_coerce_process_output(exc.stderr),
            exit_code=None,
            timed_out=True,
        )

    return BashExecution(
        command=command,
        cwd=str(cwd),
        timeout_seconds=timeout_seconds,
        started_at=started_at,
        duration_seconds=time.monotonic() - started_monotonic,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.returncode,
    )


@dataclass(frozen=True, slots=True)
class BashTool:
    """Run commands with bash and return a normalized tool result."""

    workspace_root: Path
    default_timeout_seconds: float | None = None

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Bash",
            description=(
                "Run a shell command in a working directory and capture stdout, "
                "stderr, exit code, and execution metadata."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command string executed via /bin/bash -lc.",
                    },
                    "cwd": {
                        "type": "string",
                        "description": (
                            "Working directory for the command. Relative paths are "
                            "resolved from the tool workspace root."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Optional timeout in seconds.",
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            approval_category="exec",
        )

    def execute(self, arguments: JSONObject) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command:
            return ToolResult.error("Bash command must be a non-empty string.")

        cwd = self._resolve_cwd(arguments.get("cwd"))
        if cwd is None:
            return ToolResult.error("Bash cwd must be a string when provided.")
        if not cwd.exists():
            return ToolResult.error(
                f"Bash cwd does not exist: {cwd}",
                metadata={"command": command, "cwd": str(cwd)},
            )
        if not cwd.is_dir():
            return ToolResult.error(
                f"Bash cwd is not a directory: {cwd}",
                metadata={"command": command, "cwd": str(cwd)},
            )

        timeout_seconds = self._resolve_timeout(arguments.get("timeout_seconds"))
        if timeout_seconds is _INVALID_TIMEOUT:
            return ToolResult.error("Bash timeout_seconds must be a positive number.")

        execution = run_bash_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
        )
        metadata = execution.as_metadata()

        if execution.timed_out:
            return ToolResult.error(
                f"Command timed out after {timeout_seconds} seconds.",
                metadata=metadata,
            )
        if execution.exit_code != 0:
            return ToolResult.error(
                f"Command exited with code {execution.exit_code}.",
                metadata=metadata,
            )
        return ToolResult(
            output="Command completed successfully.",
            metadata=metadata,
        )

    def _resolve_cwd(self, raw_cwd: object) -> Path | None:
        if raw_cwd is None:
            return self.workspace_root
        if not isinstance(raw_cwd, str):
            return None

        candidate = Path(raw_cwd)
        if candidate.is_absolute():
            return candidate
        return self.workspace_root / candidate

    def _resolve_timeout(self, raw_timeout: object) -> float | None | object:
        if raw_timeout is None:
            return self.default_timeout_seconds
        if isinstance(raw_timeout, bool):
            return _INVALID_TIMEOUT
        if isinstance(raw_timeout, int | float) and raw_timeout > 0:
            return float(raw_timeout)
        return _INVALID_TIMEOUT


def _coerce_process_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


_INVALID_TIMEOUT = object()

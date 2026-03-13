from __future__ import annotations

"""Bash tool for running terminal commands within the workspace."""

import asyncio
import time
from pathlib import Path
from typing import Any

from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import Tool, ToolSpec, ToolExecutionError, build_tool_result_content


class BashTool(Tool):
    """Run terminal commands within the workspace (policy-gated).

    Key distinction: non-zero exit code is a successful tool call, not a ToolExecutionError.
    Only invalid arguments, timeout, or policy denial should raise ToolExecutionError.
    """

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="bash",
            description=(
                "Run a terminal command in the workspace. "
                "Returns exit code, stdout, and stderr. "
                "Commands with non-zero exit codes are still successful tool calls."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Timeout in seconds (default: 30)",
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["command"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command", "")
        timeout = arguments.get("timeout_seconds") or self.runtime.bash_timeout_seconds

        if not command.strip():
            raise ToolExecutionError("Command cannot be empty.")

        # Clamp timeout
        timeout = max(1, min(timeout, 300))

        cwd = str(self.runtime.cwd.resolve())

        start_time = time.time()
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise ToolExecutionError(
                    f"Command timed out after {timeout} seconds."
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Truncate if needed (reasonable limits for model context)
            max_output = 64_000  # Per stream limit
            stdout_truncated = len(stdout_bytes) > max_output
            stderr_truncated = len(stderr_bytes) > max_output

            if stdout_truncated:
                stdout_bytes = stdout_bytes[:max_output]
                # Clean up partial UTF-8
                while stdout_bytes and (stdout_bytes[-1] & 0x80) and not (stdout_bytes[-1] & 0x40):
                    stdout_bytes = stdout_bytes[:-1]

            if stderr_truncated:
                stderr_bytes = stderr_bytes[:max_output]
                while stderr_bytes and (stderr_bytes[-1] & 0x80) and not (stderr_bytes[-1] & 0x40):
                    stderr_bytes = stderr_bytes[:-1]

            # Decode outputs, handling potential encoding issues
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            return build_tool_result_content({
                "ok": True,
                "tool": "bash",
                "command": command,
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "duration_ms": duration_ms,
            })

        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Failed to execute command: {exc}") from exc

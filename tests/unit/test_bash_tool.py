from __future__ import annotations

from pathlib import Path
import shlex
import sys

from agentlet.tools.exec.bash import BashTool, run_bash_command


def test_run_bash_command_captures_stdout_stderr_exit_code_and_cwd(
    tmp_path: Path,
) -> None:
    execution = run_bash_command(
        "printf 'hello'; printf 'warning' >&2; pwd; exit 0",
        cwd=tmp_path,
    )

    assert execution.command == "printf 'hello'; printf 'warning' >&2; pwd; exit 0"
    assert execution.cwd == str(tmp_path)
    assert execution.exit_code == 0
    assert execution.timed_out is False
    assert execution.stdout == f"hello{tmp_path}\n"
    assert execution.stderr == "warning"
    assert execution.duration_seconds >= 0


def test_bash_tool_returns_structured_success_result_for_relative_cwd(
    tmp_path: Path,
) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    tool = BashTool(workspace_root=tmp_path)

    result = tool.execute(
        {
            "command": "pwd",
            "cwd": "nested",
        }
    )

    assert result.is_error is False
    assert result.output == "Command completed successfully."
    assert result.metadata == {
        "command": "pwd",
        "cwd": str(nested_dir),
        "timeout_seconds": None,
        "started_at": result.metadata["started_at"],
        "duration_seconds": result.metadata["duration_seconds"],
        "stdout": f"{nested_dir}\n",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
    }
    assert isinstance(result.metadata["started_at"], float)
    assert isinstance(result.metadata["duration_seconds"], float)
    assert result.metadata["duration_seconds"] >= 0


def test_bash_tool_returns_error_result_for_non_zero_exit_code(
    tmp_path: Path,
) -> None:
    tool = BashTool(workspace_root=tmp_path)

    result = tool.execute(
        {
            "command": "printf 'out'; printf 'boom' >&2; exit 7",
            "cwd": ".",
        }
    )

    assert result.is_error is True
    assert result.output == "Command exited with code 7."
    assert result.metadata == {
        "command": "printf 'out'; printf 'boom' >&2; exit 7",
        "cwd": str(tmp_path),
        "timeout_seconds": None,
        "started_at": result.metadata["started_at"],
        "duration_seconds": result.metadata["duration_seconds"],
        "stdout": "out",
        "stderr": "boom",
        "exit_code": 7,
        "timed_out": False,
    }
    assert isinstance(result.metadata["started_at"], float)
    assert isinstance(result.metadata["duration_seconds"], float)
    assert result.metadata["duration_seconds"] >= 0


def test_bash_tool_returns_timeout_result(tmp_path: Path) -> None:
    tool = BashTool(workspace_root=tmp_path)
    quoted_python = shlex.quote(sys.executable)

    result = tool.execute(
        {
            "command": f"{quoted_python} -c 'import time; time.sleep(1)'",
            "timeout_seconds": 0.05,
        }
    )

    assert result.is_error is True
    assert result.output == "Command timed out after 0.05 seconds."
    assert result.metadata is not None
    assert result.metadata["command"] == (
        f"{quoted_python} -c 'import time; time.sleep(1)'"
    )
    assert result.metadata["cwd"] == str(tmp_path)
    assert result.metadata["timeout_seconds"] == 0.05
    assert result.metadata["stdout"] == ""
    assert result.metadata["stderr"] == ""
    assert result.metadata["exit_code"] is None
    assert result.metadata["timed_out"] is True
    assert isinstance(result.metadata["started_at"], float)
    assert isinstance(result.metadata["duration_seconds"], float)
    assert result.metadata["duration_seconds"] >= 0

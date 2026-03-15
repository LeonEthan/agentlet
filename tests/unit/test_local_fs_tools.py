from __future__ import annotations

import asyncio
import json

import pytest

from agentlet.agent.tools.bash import BashTool
from agentlet.agent.tools.local_fs import EditTool, GlobTool, GrepTool, ReadTool, WriteTool
from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import ToolExecutionError


def test_read_supports_line_ranges_with_metadata(tmp_path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    tool = ReadTool(ToolRuntimeConfig(cwd=tmp_path))
    result = json.loads(
        asyncio.run(tool.execute({"path": "sample.txt", "start_line": 2, "end_line": 3}))
    )

    assert result["content"] == "line2\nline3\n"
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["total_lines"] == 3
    assert result["truncated"] is False


def test_read_rejects_workspace_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")

    try:
        tool = ReadTool(ToolRuntimeConfig(cwd=tmp_path))
        with pytest.raises(ToolExecutionError, match="outside the workspace"):
            asyncio.run(tool.execute({"path": str(outside)}))
    finally:
        outside.unlink(missing_ok=True)


def test_read_truncates_utf8_without_dropping_complete_character(tmp_path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("你好世界", encoding="utf-8")

    tool = ReadTool(ToolRuntimeConfig(cwd=tmp_path, max_read_bytes=len("你好".encode("utf-8"))))
    result = json.loads(asyncio.run(tool.execute({"path": "sample.txt"})))

    assert result["content"] == "你好"
    assert result["truncated"] is True


def test_glob_skips_symlink_targets_outside_workspace(tmp_path) -> None:
    outside_path = tmp_path.parent / "agentlet-glob-outside.txt"
    outside_path.write_text("outside\n", encoding="utf-8")

    try:
        (tmp_path / "inside.txt").write_text("inside\n", encoding="utf-8")
        (tmp_path / "escape.txt").symlink_to(outside_path)

        tool = GlobTool(ToolRuntimeConfig(cwd=tmp_path))
        result = json.loads(asyncio.run(tool.execute({"pattern": "*.txt"})))

        assert result["matches"] == ["inside.txt"]
        assert result["truncated"] is False
    finally:
        outside_path.unlink(missing_ok=True)


def test_grep_skips_symlink_targets_outside_workspace(tmp_path) -> None:
    outside_path = tmp_path.parent / "agentlet-grep-outside.txt"
    outside_path.write_text("secret-token\n", encoding="utf-8")

    try:
        (tmp_path / "inside.txt").write_text("public-token\n", encoding="utf-8")
        (tmp_path / "escape.txt").symlink_to(outside_path)

        tool = GrepTool(ToolRuntimeConfig(cwd=tmp_path))
        result = json.loads(asyncio.run(tool.execute({"pattern": "token"})))

        assert result["matches"] == [
            {
                "path": "inside.txt",
                "line": 1,
                "column": 8,
                "text": "public-token",
            }
        ]
        assert result["truncated"] is False
    finally:
        outside_path.unlink(missing_ok=True)


def test_write_creates_new_file_and_rejects_overwrite(tmp_path) -> None:
    tool = WriteTool(ToolRuntimeConfig(cwd=tmp_path))

    created = json.loads(
        asyncio.run(tool.execute({"path": "notes/todo.txt", "content": "ship it\n"}))
    )

    assert created["created"] is True
    assert (tmp_path / "notes" / "todo.txt").read_text(encoding="utf-8") == "ship it\n"

    with pytest.raises(ToolExecutionError, match="File already exists"):
        asyncio.run(tool.execute({"path": "notes/todo.txt", "content": "again\n"}))


def test_edit_enforces_exact_match_and_replace_all(tmp_path) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("alpha\nalpha\n", encoding="utf-8")
    tool = EditTool(ToolRuntimeConfig(cwd=tmp_path))

    with pytest.raises(ToolExecutionError, match="found 2 times"):
        asyncio.run(
            tool.execute(
                {
                    "path": "sample.txt",
                    "edits": [{"old_text": "alpha", "new_text": "beta"}],
                }
            )
        )

    result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "path": "sample.txt",
                    "edits": [
                        {
                            "old_text": "alpha",
                            "new_text": "beta",
                            "replace_all": True,
                        }
                    ],
                }
            )
        )
    )

    assert result["applied_edits"] == 1
    assert result["total_replacements"] == 2
    assert target.read_text(encoding="utf-8") == "beta\nbeta\n"


def test_bash_returns_non_zero_exit_as_tool_result(tmp_path) -> None:
    tool = BashTool(ToolRuntimeConfig(cwd=tmp_path))

    result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "command": "printf 'hello'; printf 'warn' >&2; exit 7",
                }
            )
        )
    )

    assert result["exit_code"] == 7
    assert result["stdout"] == "hello"
    assert result["stderr"] == "warn"

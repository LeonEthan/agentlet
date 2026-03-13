from __future__ import annotations

import asyncio
import json

from agentlet.agent.tools.local_fs import GlobTool, GrepTool
from agentlet.agent.tools.policy import ToolRuntimeConfig


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

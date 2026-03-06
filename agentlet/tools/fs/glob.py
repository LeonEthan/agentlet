"""Glob tool implementation."""

from __future__ import annotations

from pathlib import Path

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult
from agentlet.tools.fs._common import (
    ensure_relative_pattern,
    normalize_workspace_root,
    relative_workspace_path,
    require_string_argument,
)


class GlobTool:
    """Return matching workspace file paths for a glob pattern."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._workspace_root = normalize_workspace_root(workspace_root)
        self._definition = ToolDefinition(
            name="Glob",
            description="List workspace file paths matching a glob pattern.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            approval_category="read_only",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, arguments: JSONObject) -> ToolResult:
        try:
            pattern = ensure_relative_pattern(
                require_string_argument(arguments, "pattern")
            )
            paths = sorted(self._matching_files(pattern))
        except ValueError as exc:
            return ToolResult.error(str(exc))
        except OSError as exc:
            return ToolResult.error(f"failed to glob pattern {arguments.get('pattern')}: {exc}")

        return ToolResult(
            output="\n".join(paths),
            metadata={"paths": paths, "count": len(paths)},
        )

    def _matching_files(self, pattern: str) -> list[str]:
        matches: list[str] = []
        for candidate in self._workspace_root.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
                matches.append(relative_workspace_path(self._workspace_root, resolved))
            except ValueError:
                continue
        return matches

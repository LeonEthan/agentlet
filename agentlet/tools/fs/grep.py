"""Grep tool implementation."""

from __future__ import annotations

import re
from pathlib import Path

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult
from agentlet.tools.fs._common import (
    ensure_relative_pattern,
    normalize_workspace_root,
    relative_workspace_path,
    require_string_argument,
)


class GrepTool:
    """Search UTF-8 text files in the workspace with a regular expression."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._workspace_root = normalize_workspace_root(workspace_root)
        self._definition = ToolDefinition(
            name="Grep",
            description="Search workspace text files with a regular expression.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path_glob": {"type": "string"},
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
            pattern = require_string_argument(arguments, "pattern")
            path_glob = arguments.get("path_glob")
            if path_glob is not None:
                if not isinstance(path_glob, str) or not path_glob:
                    raise ValueError("path_glob must be a non-empty string")
                path_glob = ensure_relative_pattern(path_glob, key="path_glob")
            regex = re.compile(pattern)
            matches, skipped_files = self._find_matches(
                regex=regex,
                path_glob=path_glob,
            )
        except ValueError as exc:
            return ToolResult.error(str(exc))
        except re.error as exc:
            return ToolResult.error(f"invalid regex pattern: {exc}")
        except OSError as exc:
            return ToolResult.error(f"failed to search workspace: {exc}")

        output_lines = [
            f"{match['path']}:{match['line_number']}:{match['snippet']}"
            for match in matches
        ]
        metadata: JSONObject = {"matches": matches, "count": len(matches)}
        if skipped_files:
            metadata["skipped_files"] = skipped_files
            metadata["skipped_count"] = len(skipped_files)
        if path_glob is not None:
            metadata["path_glob"] = path_glob
        return ToolResult(output="\n".join(output_lines), metadata=metadata)

    def _find_matches(
        self,
        *,
        regex: re.Pattern[str],
        path_glob: str | None,
    ) -> tuple[list[JSONObject], list[str]]:
        matches: list[JSONObject] = []
        skipped_files: list[str] = []
        for path in self._candidate_files(path_glob):
            relative_path = relative_workspace_path(self._workspace_root, path)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                skipped_files.append(relative_path)
                continue

            for line_number, line in enumerate(lines, start=1):
                if regex.search(line) is None:
                    continue
                matches.append(
                    {
                        "path": relative_path,
                        "line_number": line_number,
                        "snippet": line,
                    }
                )
        return matches, skipped_files

    def _candidate_files(self, path_glob: str | None) -> list[Path]:
        iterator = (
            self._workspace_root.glob(path_glob)
            if path_glob is not None
            else self._workspace_root.rglob("*")
        )
        files: list[Path] = []
        for candidate in iterator:
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
                relative_workspace_path(self._workspace_root, resolved)
            except ValueError:
                continue
            files.append(resolved)
        return sorted(files, key=lambda path: relative_workspace_path(self._workspace_root, path))

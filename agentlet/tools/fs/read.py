"""Read tool implementation."""

from __future__ import annotations

from pathlib import Path

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult
from agentlet.tools.fs._common import (
    normalize_workspace_root,
    optional_positive_int_argument,
    relative_workspace_path,
    require_string_argument,
    resolve_workspace_path,
)


class ReadTool:
    """Read UTF-8 text files from the workspace."""

    def __init__(self, workspace_root: str | Path) -> None:
        self._workspace_root = normalize_workspace_root(workspace_root)
        self._definition = ToolDefinition(
            name="Read",
            description="Read a UTF-8 text file from the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                    "max_chars": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            approval_category="read_only",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, arguments: JSONObject) -> ToolResult:
        try:
            raw_path = require_string_argument(arguments, "path")
            start_line = optional_positive_int_argument(arguments, "start_line")
            end_line = optional_positive_int_argument(arguments, "end_line")
            max_chars = optional_positive_int_argument(arguments, "max_chars")
            if start_line is not None and end_line is not None and start_line > end_line:
                raise ValueError("start_line must be <= end_line")

            path = resolve_workspace_path(self._workspace_root, raw_path)
            if not path.exists():
                return ToolResult.error(f"path not found: {raw_path}")
            if not path.is_file():
                return ToolResult.error(f"path is not a file: {raw_path}")

            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult.error(f"file is not valid UTF-8 text: {raw_path}")
        except ValueError as exc:
            return ToolResult.error(str(exc))
        except OSError as exc:
            return ToolResult.error(f"failed to read {raw_path}: {exc}")

        lines = content.splitlines(keepends=True)
        slice_start = 0 if start_line is None else start_line - 1
        slice_stop = len(lines) if end_line is None else end_line
        selected_lines = lines[slice_start:slice_stop]
        selected_content = "".join(selected_lines)

        truncated = False
        if max_chars is not None and len(selected_content) > max_chars:
            selected_content = selected_content[:max_chars]
            truncated = True

        metadata: JSONObject = {
            "path": relative_workspace_path(self._workspace_root, path),
            "total_lines": len(lines),
            "selected_line_count": len(selected_lines),
            "truncated": truncated,
        }
        if start_line is not None:
            metadata["start_line"] = start_line
        if end_line is not None:
            metadata["end_line"] = end_line
        if max_chars is not None:
            metadata["max_chars"] = max_chars

        return ToolResult(output=selected_content, metadata=metadata)

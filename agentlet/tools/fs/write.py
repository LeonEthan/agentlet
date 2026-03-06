"""Write tool implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult


@dataclass(slots=True)
class WriteTool:
    """Create files inside the workspace with explicit overwrite behavior."""

    workspace_root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Write",
            description="Create a file in the workspace. Existing files require overwrite=true.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            approval_category="mutating",
            metadata={"ui": {"group": "fs"}},
        )

    def execute(self, arguments: JSONObject) -> ToolResult:
        path_value = arguments.get("path")
        content_value = arguments.get("content")
        overwrite_value = arguments.get("overwrite", False)

        if not isinstance(path_value, str) or not path_value:
            return ToolResult.error("Write requires a non-empty string 'path'.")
        if not isinstance(content_value, str):
            return ToolResult.error("Write requires string 'content'.")
        if not isinstance(overwrite_value, bool):
            return ToolResult.error("Write requires boolean 'overwrite' when provided.")
        overwrite = overwrite_value

        try:
            target_path = _resolve_workspace_path(self.workspace_root, path_value)
        except ValueError as exc:
            return ToolResult.error(str(exc), metadata={"path": path_value})

        if target_path.exists() and target_path.is_dir():
            return ToolResult.error(
                f"Cannot write to directory: {path_value}",
                metadata={"path": _display_path(self.workspace_root, target_path)},
            )
        if target_path.exists() and not overwrite:
            return ToolResult.error(
                f"Refusing to overwrite existing file without overwrite=true: {path_value}",
                metadata={
                    "path": _display_path(self.workspace_root, target_path),
                    "reason": "file_exists",
                },
            )

        existed_before = target_path.exists()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content_value, encoding="utf-8")

        created = not existed_before
        action = "Overwrote" if existed_before else "Created"
        return ToolResult(
            output=f"{action} file: {_display_path(self.workspace_root, target_path)}",
            metadata={
                "path": _display_path(self.workspace_root, target_path),
                "bytes_written": len(content_value.encode("utf-8")),
                "created": created,
            },
        )


def _resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {raw_path}") from exc
    return resolved


def _display_path(workspace_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(workspace_root))
    except ValueError:
        return str(path)


__all__ = ["WriteTool"]

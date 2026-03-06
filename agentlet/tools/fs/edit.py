"""Edit tool implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult


@dataclass(slots=True)
class EditTool:
    """Apply exact-text replacements to existing workspace files."""

    workspace_root: Path = field(default_factory=Path.cwd)

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="Edit",
            description="Replace exact text in an existing workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
            approval_category="mutating",
            metadata={"ui": {"group": "fs"}},
        )

    def execute(self, arguments: JSONObject) -> ToolResult:
        path_value = arguments.get("path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        replace_all_value = arguments.get("replace_all", False)

        if not isinstance(path_value, str) or not path_value:
            return ToolResult.error("Edit requires a non-empty string 'path'.")
        if not isinstance(old_text, str) or not old_text:
            return ToolResult.error("Edit requires a non-empty string 'old_text'.")
        if not isinstance(new_text, str):
            return ToolResult.error("Edit requires string 'new_text'.")
        if not isinstance(replace_all_value, bool):
            return ToolResult.error("Edit requires boolean 'replace_all' when provided.")
        replace_all = replace_all_value

        try:
            target_path = _resolve_workspace_path(self.workspace_root, path_value)
        except ValueError as exc:
            return ToolResult.error(str(exc), metadata={"path": path_value})

        if not target_path.exists():
            return ToolResult.error(
                f"Cannot edit missing file: {path_value}",
                metadata={"path": _display_path(self.workspace_root, target_path)},
            )
        if target_path.is_dir():
            return ToolResult.error(
                f"Cannot edit directory: {path_value}",
                metadata={"path": _display_path(self.workspace_root, target_path)},
            )

        original_content = target_path.read_text(encoding="utf-8")
        match_count = original_content.count(old_text)
        if match_count == 0:
            return ToolResult.error(
                f"Edit context did not match file contents for: {path_value}",
                metadata={
                    "path": _display_path(self.workspace_root, target_path),
                    "reason": "context_mismatch",
                },
            )
        if match_count > 1 and not replace_all:
            return ToolResult.error(
                "Edit target is ambiguous; matched multiple locations. "
                "Pass replace_all=true to update every occurrence.",
                metadata={
                    "path": _display_path(self.workspace_root, target_path),
                    "match_count": match_count,
                    "reason": "ambiguous_match",
                },
            )

        updated_content = (
            original_content.replace(old_text, new_text)
            if replace_all
            else original_content.replace(old_text, new_text, 1)
        )
        replacement_count = match_count if replace_all else 1
        target_path.write_text(updated_content, encoding="utf-8")

        return ToolResult(
            output=(
                f"Updated file: {_display_path(self.workspace_root, target_path)} "
                f"({replacement_count} replacement"
                f"{'' if replacement_count == 1 else 's'})"
            ),
            metadata={
                "path": _display_path(self.workspace_root, target_path),
                "replacements": replacement_count,
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


__all__ = ["EditTool"]

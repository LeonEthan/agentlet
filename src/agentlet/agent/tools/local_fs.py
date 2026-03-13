from __future__ import annotations

"""Local filesystem tools: Read, Glob, Grep, Write, Edit."""

import re
from pathlib import Path
from typing import Any

from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import Tool, ToolSpec, ToolExecutionError, build_tool_result_content


def _resolve_workspace_path(path_str: str, cwd: Path) -> Path:
    """Resolve a path string to an absolute path within the workspace.

    Raises ToolExecutionError if the path escapes the workspace.
    """
    # Handle absolute paths that are within cwd
    input_path = Path(path_str)
    if input_path.is_absolute():
        resolved = input_path.resolve()
    else:
        resolved = (cwd / input_path).resolve()

    # Security check: path must be under cwd
    try:
        resolved.relative_to(cwd.resolve())
    except ValueError:
        raise ToolExecutionError(
            f"Path '{path_str}' resolves outside the workspace. "
            f"All paths must be within the working directory."
        )

    return resolved


def _is_text_file(path: Path) -> bool:
    """Check if a file appears to be a text file by reading a sample."""
    try:
        with open(path, "rb") as f:
            sample = f.read(8192)
        if not sample:
            return True  # Empty files are text
        # Check for null bytes (common in binary files)
        if b"\x00" in sample:
            return False
        # Try decoding as UTF-8
        sample.decode("utf-8", errors="strict")
        return True
    except (OSError, UnicodeDecodeError):
        return False


def _truncate_content(content: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate content to fit within max_bytes when UTF-8 encoded.

    Returns (truncated_content, was_truncated).
    """
    encoded = content.encode("utf-8")
    if len(encoded) <= max_bytes:
        return content, False

    # Truncate to max_bytes, then clean up partial UTF-8 sequences
    truncated_bytes = encoded[:max_bytes]
    # Remove potential partial UTF-8 character at the end
    while truncated_bytes and (truncated_bytes[-1] & 0x80) and not (truncated_bytes[-1] & 0x40):
        truncated_bytes = truncated_bytes[:-1]

    # Remove the leading continuation byte marker if present
    if truncated_bytes and (truncated_bytes[-1] & 0x80):
        truncated_bytes = truncated_bytes[:-1]

    return truncated_bytes.decode("utf-8", errors="ignore"), True


class ReadTool(Tool):
    """Read text files within the workspace, with optional line range support."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read",
            description="Read a text file within the workspace. Supports optional line ranges.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to workspace root",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based, inclusive)",
                        "minimum": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-based, inclusive)",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        path_str = arguments.get("path", "")
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")

        resolved = _resolve_workspace_path(path_str, self.runtime.cwd)

        if not resolved.exists():
            raise ToolExecutionError(f"File not found: {path_str}")
        if not resolved.is_file():
            raise ToolExecutionError(f"Path is not a file: {path_str}")

        if not _is_text_file(resolved):
            raise ToolExecutionError(f"Binary files cannot be read: {path_str}")

        # Read the file
        try:
            content = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to read file {path_str}: {exc}") from exc

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        # Apply line range if specified
        if start_line is not None or end_line is not None:
            start_idx = (start_line or 1) - 1  # Convert to 0-based
            end_idx = (end_line or total_lines)  # End is inclusive, so keep as-is for slicing

            # Clamp indices
            start_idx = max(0, start_idx)
            end_idx = min(total_lines, end_idx)

            if start_idx >= end_idx:
                selected_lines = []
            else:
                selected_lines = lines[start_idx:end_idx]
            content = "".join(selected_lines)
            actual_start = start_idx + 1
            actual_end = min(end_idx, total_lines)
        else:
            selected_lines = lines
            actual_start = 1
            actual_end = total_lines

        # Apply byte limit
        content, truncated = _truncate_content(content, self.runtime.max_read_bytes)

        return build_tool_result_content({
            "ok": True,
            "tool": "read",
            "path": path_str,
            "content": content,
            "start_line": actual_start,
            "end_line": actual_end,
            "total_lines": total_lines,
            "truncated": truncated,
        })


class GlobTool(Tool):
    """Find files by pattern within the workspace."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="glob",
            description="Find files by glob pattern within the workspace. Returns workspace-relative paths.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', 'src/**/*.ts')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["pattern"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        pattern = arguments.get("pattern", "")
        limit = arguments.get("limit") or 50  # Default limit

        # Clamp limit to reasonable range
        limit = max(1, min(limit, 100))

        cwd_resolved = self.runtime.cwd.resolve()

        try:
            matches = []
            for path in cwd_resolved.rglob(pattern):
                if path.is_file():
                    try:
                        rel_path = path.relative_to(cwd_resolved)
                        matches.append(str(rel_path))
                        if len(matches) >= limit:
                            break
                    except ValueError:
                        pass  # Skip files outside workspace

            truncated = len(matches) >= limit

            return build_tool_result_content({
                "ok": True,
                "tool": "glob",
                "pattern": pattern,
                "matches": matches,
                "truncated": truncated,
            })
        except Exception as exc:
            raise ToolExecutionError(f"Glob failed for pattern '{pattern}': {exc}") from exc


class GrepTool(Tool):
    """Search file contents with regex within the workspace."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep",
            description="Search file contents with regex pattern. Returns matches with path, line, column, and text.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g., '*.py')",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matches to return",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search is case-sensitive",
                        "default": False,
                    },
                },
                "required": ["pattern"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        pattern_str = arguments.get("pattern", "")
        glob_pattern = arguments.get("glob")
        limit = arguments.get("limit") or 20  # Default limit
        case_sensitive = arguments.get("case_sensitive", False)

        # Clamp limit
        limit = max(1, min(limit, 100))

        # Compile regex
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            regex = re.compile(pattern_str, flags)
        except re.error as exc:
            raise ToolExecutionError(f"Invalid regex pattern: {exc}") from exc

        cwd_resolved = self.runtime.cwd.resolve()
        matches = []

        try:
            # Collect files to search
            if glob_pattern:
                files = list(cwd_resolved.rglob(glob_pattern))
            else:
                files = list(cwd_resolved.rglob("*"))

            for file_path in files:
                if not file_path.is_file():
                    continue
                if not _is_text_file(file_path):
                    continue

                try:
                    rel_path = file_path.relative_to(cwd_resolved)
                except ValueError:
                    continue  # Skip files outside workspace

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()

                    for line_num, line in enumerate(lines, start=1):
                        for match in regex.finditer(line):
                            matches.append({
                                "path": str(rel_path),
                                "line": line_num,
                                "column": match.start() + 1,  # 1-based column
                                "text": line[:200],  # Limit text length per match
                            })
                            if len(matches) >= limit:
                                break
                        if len(matches) >= limit:
                            break

                    if len(matches) >= limit:
                        break

                except OSError:
                    continue  # Skip files we can't read

        except Exception as exc:
            raise ToolExecutionError(f"Grep failed: {exc}") from exc

        return build_tool_result_content({
            "ok": True,
            "tool": "grep",
            "pattern": pattern_str,
            "matches": matches,
            "truncated": len(matches) >= limit,
        })


class WriteTool(Tool):
    """Create new files within the workspace (policy-gated)."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write",
            description="Create a new file within the workspace. Fails if the file already exists.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to create, relative to workspace root",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                    "create_parents": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist",
                        "default": True,
                    },
                },
                "required": ["path", "content"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        path_str = arguments.get("path", "")
        content = arguments.get("content", "")
        create_parents = arguments.get("create_parents", True)

        resolved = _resolve_workspace_path(path_str, self.runtime.cwd)

        # Check if file already exists
        if resolved.exists():
            raise ToolExecutionError(
                f"File already exists: {path_str}. Use Edit to modify existing files."
            )

        # Check payload size
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self.runtime.max_write_bytes:
            raise ToolExecutionError(
                f"Content exceeds maximum size of {self.runtime.max_write_bytes} bytes."
            )

        try:
            if create_parents:
                resolved.parent.mkdir(parents=True, exist_ok=True)

            resolved.write_text(content, encoding="utf-8")

            return build_tool_result_content({
                "ok": True,
                "tool": "write",
                "path": path_str,
                "bytes_written": len(content_bytes),
                "created": True,
            })
        except OSError as exc:
            raise ToolExecutionError(f"Failed to write file {path_str}: {exc}") from exc


class EditTool(Tool):
    """Make precise string replacements in existing files (policy-gated)."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit",
            description=(
                "Make precise string replacements in an existing file. "
                "Each edit operation specifies old_text to find and new_text to replace it with. "
                "Edits are applied sequentially against the latest buffer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit, relative to workspace root",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of edit operations to apply",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": "Text to find and replace",
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement text",
                                },
                                "replace_all": {
                                    "type": "boolean",
                                    "description": "Replace all occurrences (default: false)",
                                    "default": False,
                                },
                            },
                            "required": ["old_text", "new_text"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        path_str = arguments.get("path", "")
        edits = arguments.get("edits", [])

        if not isinstance(edits, list):
            raise ToolExecutionError("'edits' must be a list of edit operations.")

        if not edits:
            raise ToolExecutionError("At least one edit operation is required.")

        resolved = _resolve_workspace_path(path_str, self.runtime.cwd)

        if not resolved.exists():
            raise ToolExecutionError(f"File not found: {path_str}")
        if not resolved.is_file():
            raise ToolExecutionError(f"Path is not a file: {path_str}")

        if not _is_text_file(resolved):
            raise ToolExecutionError(f"Binary files cannot be edited: {path_str}")

        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to read file {path_str}: {exc}") from exc

        total_replacements = 0
        applied_edits = 0

        for i, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise ToolExecutionError(f"Edit {i+1} must be an object.")

            old_text = edit.get("old_text")
            new_text = edit.get("new_text")
            replace_all = edit.get("replace_all", False)

            if old_text is None:
                raise ToolExecutionError(f"Edit {i+1}: 'old_text' is required.")

            # Count occurrences
            count = content.count(old_text)

            if count == 0:
                raise ToolExecutionError(
                    f"Edit {i+1}: 'old_text' not found in file."
                )

            if not replace_all and count > 1:
                raise ToolExecutionError(
                    f"Edit {i+1}: 'old_text' found {count} times. "
                    f"Use replace_all=true to replace all occurrences."
                )

            # Apply replacement
            if replace_all:
                content = content.replace(old_text, new_text)
                total_replacements += count
            else:
                content = content.replace(old_text, new_text, 1)
                total_replacements += 1

            applied_edits += 1

        # Check final size
        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self.runtime.max_write_bytes:
            raise ToolExecutionError(
                f"Resulting file exceeds maximum size of {self.runtime.max_write_bytes} bytes."
            )

        # Write the modified content
        try:
            resolved.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise ToolExecutionError(f"Failed to write file {path_str}: {exc}") from exc

        return build_tool_result_content({
            "ok": True,
            "tool": "edit",
            "path": path_str,
            "applied_edits": applied_edits,
            "total_replacements": total_replacements,
            "bytes_written": len(content_bytes),
        })

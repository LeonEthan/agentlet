"""Shared helpers for filesystem tools."""

from __future__ import annotations

from pathlib import Path

from agentlet.core.types import JSONObject


def normalize_workspace_root(workspace_root: str | Path) -> Path:
    """Resolve and normalize the workspace root once during tool setup."""

    return Path(workspace_root).expanduser().resolve()


def require_string_argument(arguments: JSONObject, key: str) -> str:
    """Return a required non-empty string argument."""

    value = arguments.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def optional_positive_int_argument(arguments: JSONObject, key: str) -> int | None:
    """Return an optional integer argument constrained to positive values."""

    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return value


def ensure_relative_pattern(pattern: str, key: str = "pattern") -> str:
    """Reject absolute glob-like patterns so search stays rooted in the workspace."""

    if Path(pattern).is_absolute():
        raise ValueError(f"{key} must be relative to the workspace root")
    return pattern


def resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    """Resolve a user path and ensure it stays inside the workspace root."""

    candidate = (workspace_root / raw_path).expanduser().resolve()
    try:
        candidate.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError("path must stay inside the workspace root") from exc
    return candidate


def relative_workspace_path(workspace_root: Path, path: Path) -> str:
    """Convert a resolved workspace path into a stable relative string."""

    return path.relative_to(workspace_root).as_posix()

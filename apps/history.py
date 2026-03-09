"""Persistent input history for the agentlet CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator


class InputHistory:
    """Persistent input history with file-backed storage."""

    DEFAULT_HISTORY_FILE = "~/.agentlet/history"
    MAX_HISTORY_ENTRIES = 1000

    def __init__(self, history_file: str | Path | None = None) -> None:
        self.history_file = Path(history_file or self.DEFAULT_HISTORY_FILE).expanduser()
        self._entries: list[str] = []
        self._load()

    def _load(self) -> None:
        """Load history from file."""
        if self.history_file.exists():
            try:
                content = self.history_file.read_text(encoding="utf-8")
                self._entries = [
                    line.rstrip("\n")
                    for line in content.split("\n")
                    if line.strip()
                ][-self.MAX_HISTORY_ENTRIES:]
            except (IOError, OSError):
                self._entries = []

    def _save(self) -> None:
        """Save history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(self._entries) + "\n"
            self.history_file.write_text(content, encoding="utf-8")
        except (IOError, OSError):
            pass

    def add(self, entry: str) -> None:
        """Add an entry to the history."""
        entry = entry.strip()
        if not entry:
            return

        # Don't add duplicate consecutive entries
        if self._entries and self._entries[-1] == entry:
            return

        self._entries.append(entry)

        # Trim to max size
        if len(self._entries) > self.MAX_HISTORY_ENTRIES:
            self._entries = self._entries[-self.MAX_HISTORY_ENTRIES:]

        self._save()

    def get_entries(self) -> list[str]:
        """Get all history entries."""
        return list(self._entries)

    def __iter__(self) -> Iterator[str]:
        """Iterate over history entries."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Get the number of history entries."""
        return len(self._entries)

    def clear(self) -> None:
        """Clear all history."""
        self._entries = []
        self._save()


__all__ = ["InputHistory"]

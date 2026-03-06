"""Markdown-backed durable memory."""

from __future__ import annotations

from pathlib import Path


class MemoryStore:
    """Persist durable memory as a plain markdown file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read(self) -> str:
        """Return the stored markdown content or an empty string when absent."""

        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def write(self, content: str) -> None:
        """Replace the durable memory file with new markdown content."""

        if not isinstance(content, str):
            raise TypeError("memory content must be a string")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content, encoding="utf-8")

from __future__ import annotations

from pathlib import Path

import pytest

from agentlet.tools.fs.glob import GlobTool
from agentlet.tools.fs.grep import GrepTool
from agentlet.tools.fs.read import ReadTool


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_read_tool_reads_selected_lines_and_truncates(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "notes.txt",
        "alpha\nbeta\ngamma\ndelta\n",
    )
    tool = ReadTool(tmp_path)

    result = tool.execute(
        {
            "path": "notes.txt",
            "start_line": 2,
            "end_line": 4,
            "max_chars": 8,
        }
    )

    assert result.is_error is False
    assert result.output == "beta\ngam"
    assert result.metadata == {
        "path": "notes.txt",
        "total_lines": 4,
        "selected_line_count": 3,
        "truncated": True,
        "start_line": 2,
        "end_line": 4,
        "max_chars": 8,
    }


def test_read_tool_returns_error_for_invalid_range(tmp_path: Path) -> None:
    _write_file(tmp_path / "notes.txt", "alpha\nbeta\n")
    tool = ReadTool(tmp_path)

    result = tool.execute(
        {
            "path": "notes.txt",
            "start_line": 3,
            "end_line": 2,
        }
    )

    assert result.is_error is True
    assert result.output == "start_line must be <= end_line"


def test_read_tool_returns_error_for_missing_file(tmp_path: Path) -> None:
    tool = ReadTool(tmp_path)

    result = tool.execute({"path": "missing.txt"})

    assert result.is_error is True
    assert result.output == "path not found: missing.txt"


def test_read_tool_returns_normalized_error_for_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"\xff\xfe\x00\x01")
    tool = ReadTool(tmp_path)

    result = tool.execute({"path": "data.bin"})

    assert result.is_error is True
    assert result.output == "file is not valid UTF-8 text: data.bin"


def test_glob_tool_returns_sorted_file_paths_only(tmp_path: Path) -> None:
    _write_file(tmp_path / "src" / "b.py", "print('b')\n")
    _write_file(tmp_path / "src" / "a.py", "print('a')\n")
    _write_file(tmp_path / "src" / "nested" / "c.py", "print('c')\n")
    (tmp_path / "src" / "empty_dir").mkdir(parents=True)
    tool = GlobTool(tmp_path)

    result = tool.execute({"pattern": "src/**/*.py"})

    assert result.is_error is False
    assert result.output == "src/a.py\nsrc/b.py\nsrc/nested/c.py"
    assert result.metadata == {
        "paths": ["src/a.py", "src/b.py", "src/nested/c.py"],
        "count": 3,
    }


def test_glob_tool_returns_error_for_invalid_pattern_argument(tmp_path: Path) -> None:
    tool = GlobTool(tmp_path)

    result = tool.execute({"pattern": 123})  # type: ignore[arg-type]

    assert result.is_error is True
    assert result.output == "pattern must be a non-empty string"


def test_grep_tool_returns_match_locations_and_snippets(tmp_path: Path) -> None:
    _write_file(
        tmp_path / "pkg" / "alpha.py",
        "hello world\nskip me\nhello again\n",
    )
    _write_file(
        tmp_path / "pkg" / "beta.py",
        "nothing here\nsay hello\n",
    )
    _write_file(tmp_path / "pkg" / "notes.txt", "hello from txt\n")
    tool = GrepTool(tmp_path)

    result = tool.execute(
        {
            "pattern": "hello",
            "path_glob": "pkg/**/*.py",
        }
    )

    assert result.is_error is False
    assert result.output == (
        "pkg/alpha.py:1:hello world\n"
        "pkg/alpha.py:3:hello again\n"
        "pkg/beta.py:2:say hello"
    )
    assert result.metadata == {
        "matches": [
            {"path": "pkg/alpha.py", "line_number": 1, "snippet": "hello world"},
            {"path": "pkg/alpha.py", "line_number": 3, "snippet": "hello again"},
            {"path": "pkg/beta.py", "line_number": 2, "snippet": "say hello"},
        ],
        "count": 3,
        "path_glob": "pkg/**/*.py",
    }


def test_grep_tool_returns_empty_result_when_nothing_matches(tmp_path: Path) -> None:
    _write_file(tmp_path / "pkg" / "alpha.py", "print('alpha')\n")
    tool = GrepTool(tmp_path)

    result = tool.execute({"pattern": "hello"})

    assert result.is_error is False
    assert result.output == ""
    assert result.metadata == {"matches": [], "count": 0}


def test_grep_tool_returns_error_for_invalid_regex(tmp_path: Path) -> None:
    _write_file(tmp_path / "pkg" / "alpha.py", "print('alpha')\n")
    tool = GrepTool(tmp_path)

    result = tool.execute({"pattern": "["})

    assert result.is_error is True
    assert result.output.startswith("invalid regex pattern:")


def test_grep_tool_skips_unreadable_files_and_returns_other_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readable_path = tmp_path / "pkg" / "alpha.py"
    unreadable_path = tmp_path / "pkg" / "broken.py"
    _write_file(readable_path, "hello world\n")
    _write_file(unreadable_path, "hello hidden\n")
    tool = GrepTool(tmp_path)
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
        if self == unreadable_path:
            raise PermissionError("permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    result = tool.execute({"pattern": "hello", "path_glob": "pkg/*.py"})

    assert result.is_error is False
    assert result.output == "pkg/alpha.py:1:hello world"
    assert result.metadata == {
        "matches": [
            {"path": "pkg/alpha.py", "line_number": 1, "snippet": "hello world"},
        ],
        "count": 1,
        "skipped_files": ["pkg/broken.py"],
        "skipped_count": 1,
        "path_glob": "pkg/*.py",
    }

from __future__ import annotations

from agentlet.tools.fs.edit import EditTool
from agentlet.tools.fs.write import WriteTool


def test_write_creates_new_file(tmp_path) -> None:
    tool = WriteTool(workspace_root=tmp_path)

    result = tool.execute({"path": "notes/todo.txt", "content": "ship it\n"})

    assert result.is_error is False
    assert result.output == "Created file: notes/todo.txt"
    assert result.metadata == {
        "path": "notes/todo.txt",
        "bytes_written": len("ship it\n".encode("utf-8")),
        "created": True,
    }
    assert (tmp_path / "notes" / "todo.txt").read_text(encoding="utf-8") == "ship it\n"


def test_write_rejects_overwrite_without_flag(tmp_path) -> None:
    tool = WriteTool(workspace_root=tmp_path)
    target = tmp_path / "README.md"
    target.write_text("original\n", encoding="utf-8")

    result = tool.execute({"path": "README.md", "content": "updated\n"})

    assert result.is_error is True
    assert "overwrite=true" in result.output
    assert result.metadata == {"path": "README.md", "reason": "file_exists"}
    assert target.read_text(encoding="utf-8") == "original\n"


def test_write_allows_overwrite_with_flag(tmp_path) -> None:
    tool = WriteTool(workspace_root=tmp_path)
    target = tmp_path / "README.md"
    target.write_text("original\n", encoding="utf-8")

    result = tool.execute(
        {"path": "README.md", "content": "updated\n", "overwrite": True}
    )

    assert result.is_error is False
    assert result.output == "Overwrote file: README.md"
    assert result.metadata == {
        "path": "README.md",
        "bytes_written": len("updated\n".encode("utf-8")),
        "created": False,
    }
    assert target.read_text(encoding="utf-8") == "updated\n"


def test_edit_replaces_exact_match(tmp_path) -> None:
    tool = EditTool(workspace_root=tmp_path)
    target = tmp_path / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    result = tool.execute(
        {
            "path": "app.py",
            "old_text": "print('hello')",
            "new_text": "print('goodbye')",
        }
    )

    assert result.is_error is False
    assert result.output == "Updated file: app.py (1 replacement)"
    assert result.metadata == {"path": "app.py", "replacements": 1}
    assert target.read_text(encoding="utf-8") == "print('goodbye')\n"


def test_edit_fails_when_context_does_not_match(tmp_path) -> None:
    tool = EditTool(workspace_root=tmp_path)
    target = tmp_path / "app.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    result = tool.execute(
        {
            "path": "app.py",
            "old_text": "print('missing')",
            "new_text": "print('goodbye')",
        }
    )

    assert result.is_error is True
    assert "did not match" in result.output
    assert result.metadata == {"path": "app.py", "reason": "context_mismatch"}
    assert target.read_text(encoding="utf-8") == "print('hello')\n"


def test_edit_fails_when_exact_target_is_ambiguous(tmp_path) -> None:
    tool = EditTool(workspace_root=tmp_path)
    target = tmp_path / "app.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")

    result = tool.execute(
        {
            "path": "app.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
        }
    )

    assert result.is_error is True
    assert "ambiguous" in result.output
    assert result.metadata == {
        "path": "app.py",
        "match_count": 2,
        "reason": "ambiguous_match",
    }
    assert target.read_text(encoding="utf-8") == "value = 1\nvalue = 1\n"


def test_edit_rejects_missing_file(tmp_path) -> None:
    tool = EditTool(workspace_root=tmp_path)

    result = tool.execute(
        {
            "path": "missing.py",
            "old_text": "before",
            "new_text": "after",
        }
    )

    assert result.is_error is True
    assert result.output == "Cannot edit missing file: missing.py"
    assert result.metadata == {"path": "missing.py"}

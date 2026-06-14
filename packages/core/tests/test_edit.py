from __future__ import annotations

from pathlib import Path

import pytest

from clawde_core.tools import edit as edit_module
from clawde_core.tools._files import FileToolError
from clawde_core.tools.edit import EditTool


def test_edit_replaces_a_unique_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("a = 1\nb = 2\n", encoding="utf-8")

    result = EditTool().invoke({"path": str(target), "old_string": "a = 1", "new_string": "a = 10"})

    assert target.read_text(encoding="utf-8") == "a = 10\nb = 2\n"
    assert result == f"Edited {target}: replaced 1 occurrence at line 1."


def test_edit_replace_all_replaces_every_occurrence(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x\nx\n", encoding="utf-8")

    result = EditTool().invoke(
        {"path": str(target), "old_string": "x", "new_string": "y", "replace_all": True}
    )

    assert target.read_text(encoding="utf-8") == "y\ny\n"
    assert result == f"Edited {target}: replaced 2 occurrences."


def test_edit_rejects_an_ambiguous_match(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("x\nx\n", encoding="utf-8")

    result = EditTool().invoke({"path": str(target), "old_string": "x", "new_string": "y"})

    assert result.startswith("Error:")
    assert "occurs 2 times" in result
    assert target.read_text(encoding="utf-8") == "x\nx\n"  # unchanged


def test_edit_reports_old_string_not_found(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("abc\n", encoding="utf-8")

    result = EditTool().invoke({"path": str(target), "old_string": "zzz", "new_string": "y"})

    assert result.startswith("Error:")
    assert "not found" in result


def test_edit_reports_a_missing_file(tmp_path: Path) -> None:
    result = EditTool().invoke(
        {"path": str(tmp_path / "nope.txt"), "old_string": "a", "new_string": "b"}
    )

    assert result.startswith("Error:")
    assert "File not found" in result


def test_edit_rejects_an_empty_old_string(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("abc\n", encoding="utf-8")

    result = EditTool().invoke({"path": str(target), "old_string": "", "new_string": "x"})

    assert result.startswith("Error:")
    assert "write" in result


def test_edit_rejects_a_no_op_edit(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("abc\n", encoding="utf-8")

    result = EditTool().invoke({"path": str(target), "old_string": "abc", "new_string": "abc"})

    assert result.startswith("Error:")
    assert "identical" in result


def test_edit_reports_a_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "f.txt"
    target.write_text("a = 1\n", encoding="utf-8")

    def _deny(path: str, content: str) -> int:
        raise FileToolError("disk full")

    monkeypatch.setattr(edit_module, "write_text", _deny)
    result = EditTool().invoke({"path": str(target), "old_string": "a = 1", "new_string": "a = 2"})

    assert result == "Error: disk full"

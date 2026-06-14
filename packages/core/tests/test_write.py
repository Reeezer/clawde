from __future__ import annotations

from pathlib import Path

from clawde_core.tools.write import WriteTool


def test_write_creates_a_file_and_confirms(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    result = WriteTool().invoke({"path": str(target), "content": "hello\nworld"})

    assert target.read_text(encoding="utf-8") == "hello\nworld"
    assert result == f"Wrote 11 bytes (2 lines) to {target}."


def test_write_overwrites_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old content", encoding="utf-8")

    WriteTool().invoke({"path": str(target), "content": "new"})

    assert target.read_text(encoding="utf-8") == "new"


def test_write_reports_zero_lines_for_empty_content(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"

    result = WriteTool().invoke({"path": str(target), "content": ""})

    assert result == f"Wrote 0 bytes (0 lines) to {target}."
    assert target.read_text(encoding="utf-8") == ""


def test_write_reports_a_missing_parent_directory(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "out.txt"

    result = WriteTool().invoke({"path": str(target), "content": "x"})

    assert result.startswith("Error:")
    assert "Parent directory does not exist" in result
    assert not target.exists()

from __future__ import annotations

from pathlib import Path

from clawde_core.tools.read import ReadTool


def test_read_returns_content_with_line_numbers(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    result = ReadTool().invoke({"path": str(target)})

    assert result == "     1\talpha\n     2\tbeta\n     3\tgamma"


def test_read_applies_offset_and_limit(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("l1\nl2\nl3\nl4\nl5\n", encoding="utf-8")

    result = ReadTool().invoke({"path": str(target), "offset": 2, "limit": 2})

    assert result == "     2\tl2\n     3\tl3"


def test_read_reports_a_missing_file(tmp_path: Path) -> None:
    result = ReadTool().invoke({"path": str(tmp_path / "nope.txt")})

    assert result.startswith("Error:")
    assert "not found" in result.lower()


def test_read_reports_an_empty_file(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    target.write_text("", encoding="utf-8")

    assert ReadTool().invoke({"path": str(target)}) == "(empty file)"


def test_read_reports_offset_past_end_of_file(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("only one line\n", encoding="utf-8")

    result = ReadTool().invoke({"path": str(target), "offset": 5})

    assert result.startswith("Error:")
    assert "past the end" in result

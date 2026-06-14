from __future__ import annotations

from pathlib import Path

import pytest

from clawde_core.tools import grep as grep_module
from clawde_core.tools.grep import GrepTool


def test_grep_returns_matching_lines_with_numbers(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_text("alpha\nbeta\nalpha2\n", encoding="utf-8")

    result = GrepTool().invoke({"pattern": "alpha", "path": str(tmp_path)})

    posix = target.as_posix()
    assert result == f"{posix}:1:alpha\n{posix}:3:alpha2"


def test_grep_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "text.txt").write_text("match here\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"\x00match")  # contains the term but is binary

    result = GrepTool().invoke({"pattern": "match", "path": str(tmp_path)})

    assert "text.txt:1:match here" in result
    assert "blob.bin" not in result


def test_grep_reports_no_matches(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("nothing here\n", encoding="utf-8")

    result = GrepTool().invoke({"pattern": "zzz", "path": str(tmp_path)})

    assert result == "No matches for 'zzz'."


def test_grep_reports_a_missing_base_directory(tmp_path: Path) -> None:
    result = GrepTool().invoke({"pattern": "x", "path": str(tmp_path / "nope")})

    assert result.startswith("Error:")
    assert "base directory not found" in result


def test_grep_reports_an_invalid_regex(tmp_path: Path) -> None:
    result = GrepTool().invoke({"pattern": "[", "path": str(tmp_path)})

    assert result.startswith("Error:")
    assert "invalid regular expression" in result


def test_grep_reports_an_invalid_include_pattern(tmp_path: Path) -> None:
    result = GrepTool().invoke({"pattern": "x", "path": str(tmp_path), "include": ""})

    assert result.startswith("Error:")
    assert "invalid include pattern" in result


def test_grep_stops_at_the_match_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "f.txt").write_text("x\nx\nx\n", encoding="utf-8")
    monkeypatch.setattr(grep_module, "_MAX_MATCHES", 1)

    result = GrepTool().invoke({"pattern": "x", "path": str(tmp_path)})

    assert result.splitlines()[-1] == "... (stopped at 1 matches)"

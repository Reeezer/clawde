from __future__ import annotations

from pathlib import Path

import pytest

from clawde_core.tools import glob as glob_module
from clawde_core.tools.glob import GlobTool


def test_glob_lists_matching_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.py").write_text("", encoding="utf-8")
    (tmp_path / "c.txt").write_text("", encoding="utf-8")

    result = GlobTool().invoke({"pattern": "*.py", "path": str(tmp_path)})

    assert result == f"{(tmp_path / 'a.py').as_posix()}\n{(tmp_path / 'b.py').as_posix()}"


def test_glob_reports_no_matches(tmp_path: Path) -> None:
    result = GlobTool().invoke({"pattern": "*.md", "path": str(tmp_path)})

    assert "No files match" in result


def test_glob_reports_a_missing_base_directory(tmp_path: Path) -> None:
    result = GlobTool().invoke({"pattern": "*.py", "path": str(tmp_path / "nope")})

    assert result.startswith("Error:")
    assert "base directory not found" in result


def test_glob_reports_an_invalid_pattern(tmp_path: Path) -> None:
    result = GlobTool().invoke({"pattern": "", "path": str(tmp_path)})

    assert result.startswith("Error:")
    assert "invalid glob pattern" in result


def test_glob_truncates_long_result_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(glob_module, "_MAX_RESULTS", 1)

    result = GlobTool().invoke({"pattern": "*.py", "path": str(tmp_path)})

    lines = result.splitlines()
    assert lines[0] == (tmp_path / "a.py").as_posix()
    assert lines[1] == "... (2 more not shown)"

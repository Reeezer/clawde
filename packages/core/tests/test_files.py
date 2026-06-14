from __future__ import annotations

from pathlib import Path

import pytest

from clawde_core.tools._files import FileToolError, read_text, write_text


def test_read_text_returns_utf8_content(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    target.write_bytes(b"alpha\nbeta\n")  # raw bytes: read_text returns content faithfully

    assert read_text(str(target)) == "alpha\nbeta\n"


def test_read_text_errors_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileToolError, match="File not found"):
        read_text(str(tmp_path / "nope.txt"))


def test_read_text_errors_on_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileToolError, match="Not a file"):
        read_text(str(tmp_path))


def test_read_text_errors_when_the_file_is_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "f.txt"
    target.write_text("data", encoding="utf-8")

    def _deny(self: Path) -> bytes:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", _deny)
    with pytest.raises(FileToolError, match="Could not read"):
        read_text(str(target))


def test_read_text_errors_when_over_the_size_cap(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    target.write_text("123456", encoding="utf-8")

    with pytest.raises(FileToolError, match="too large"):
        read_text(str(target), max_bytes=2)


def test_read_text_errors_on_nul_bytes(tmp_path: Path) -> None:
    target = tmp_path / "b.bin"
    target.write_bytes(b"a\x00b")

    with pytest.raises(FileToolError, match="binary"):
        read_text(str(target))


def test_read_text_errors_on_invalid_utf8(tmp_path: Path) -> None:
    target = tmp_path / "u.bin"
    target.write_bytes(b"\xff\xfe")  # invalid UTF-8, no NUL byte

    with pytest.raises(FileToolError, match="not valid UTF-8"):
        read_text(str(target))


def test_write_text_creates_a_file_and_returns_byte_count(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"

    written = write_text(str(target), "héllo")  # 'é' is two UTF-8 bytes

    assert written == 6
    assert target.read_text(encoding="utf-8") == "héllo"


def test_write_text_errors_on_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileToolError, match="Is a directory"):
        write_text(str(tmp_path), "x")


def test_write_text_errors_when_parent_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileToolError, match="Parent directory does not exist"):
        write_text(str(tmp_path / "missing" / "out.txt"), "x")


def test_write_text_errors_when_the_file_is_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _deny(self: Path, data: bytes) -> int:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_bytes", _deny)
    with pytest.raises(FileToolError, match="Could not write"):
        write_text(str(tmp_path / "out.txt"), "x")

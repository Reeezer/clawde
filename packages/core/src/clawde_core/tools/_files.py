"""Shared filesystem helpers for the file tools (``read``, ``write``, ``edit``, ``grep``).

Centralises the boundary checks every file tool needs — existence, file-vs-dir,
a size cap, and UTF-8/binary detection — so they fail the same friendly way
instead of each re-deriving it. Tools that must report the problem (``read``,
``edit``) surface :class:`FileToolError`'s text; tools that prefer to skip
unreadable files (``grep``) catch it and move on.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_MAX_BYTES = 256 * 1024


class FileToolError(Exception):
    """A path cannot be used as a UTF-8 text file (missing, a directory, too big, binary)."""


def read_text(path: str, *, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Read ``path`` as UTF-8 text, raising :class:`FileToolError` on any problem."""
    target = Path(path)
    if not target.exists():
        raise FileToolError(f"File not found: {path}")
    if not target.is_file():
        raise FileToolError(f"Not a file: {path}")
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise FileToolError(f"Could not read {path}: {exc}") from exc
    if len(data) > max_bytes:
        raise FileToolError(f"File too large ({len(data)} bytes; limit {max_bytes}): {path}")
    if b"\x00" in data:
        raise FileToolError(f"File appears to be binary (contains NUL bytes): {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FileToolError(f"File is not valid UTF-8 text: {path}") from exc


def write_text(path: str, content: str) -> int:
    """Write ``content`` to ``path`` as UTF-8 (parent must exist); return bytes written."""
    target = Path(path)
    if target.is_dir():
        raise FileToolError(f"Is a directory, not a file: {path}")
    if not target.parent.exists():
        raise FileToolError(f"Parent directory does not exist: {target.parent}")
    data = content.encode("utf-8")
    try:
        target.write_bytes(data)
    except OSError as exc:
        raise FileToolError(f"Could not write {path}: {exc}") from exc
    return len(data)

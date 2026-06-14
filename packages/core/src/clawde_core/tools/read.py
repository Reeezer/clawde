"""The ``read`` tool — read a UTF-8 text file with 1-based line numbers.

Numbered output (``cat -n`` style) is the contract the model and the ``edit``
tool rely on: line numbers let the model point at exact lines. ``offset``/``limit``
read a window of a large file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from clawde_core.models import ToolSpec
from clawde_core.tools._files import FileToolError, read_text
from clawde_core.tools.base import Tool
from clawde_core.tools.registry import TOOLS

_SPEC = ToolSpec(
    name="read",
    description=(
        "Read a UTF-8 text file and return its contents with 1-based line numbers "
        "(like `cat -n`). Use `offset` and `limit` to read a window of a large file. "
        "Returns a friendly error for a missing, binary, or oversized file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "minimum": 1,
                "description": "1-based line number to start from (default 1).",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum number of lines to return (default: to end of file).",
            },
        },
        "required": ["path"],
    },
)


@TOOLS.register("read")
def _build_read() -> Tool:
    return ReadTool()


class ReadTool(Tool):
    """Reads a text file and returns it with line numbers."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        path = str(arguments["path"])
        try:
            text = read_text(path)
        except FileToolError as exc:
            return f"Error: {exc}"
        offset = cast(int, arguments.get("offset", 1))
        limit_arg = arguments.get("limit")
        limit = cast(int, limit_arg) if limit_arg is not None else None
        return _with_line_numbers(text, offset=offset, limit=limit)


def _with_line_numbers(text: str, *, offset: int, limit: int | None) -> str:
    lines = text.splitlines()
    if not lines:
        return "(empty file)"
    if offset > len(lines):
        return f"Error: offset {offset} is past the end of the file ({len(lines)} lines)."
    end = len(lines) if limit is None else min(len(lines), offset - 1 + limit)
    window = lines[offset - 1 : end]
    return "\n".join(f"{offset + i:>6}\t{line}" for i, line in enumerate(window))

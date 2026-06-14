"""The ``write`` tool — create or overwrite a UTF-8 text file.

Mutating, but unguarded for now: permission gating arrives in Phase 5. The parent
directory must already exist — creating directory trees is a deliberate, separate
act (use ``bash``), not a silent side effect of writing a file.
"""

from __future__ import annotations

from collections.abc import Mapping

from clawde_core.models import ToolSpec
from clawde_core.tools._files import FileToolError, write_text
from clawde_core.tools.base import Tool
from clawde_core.tools.registry import TOOLS

_SPEC = ToolSpec(
    name="write",
    description=(
        "Create a new file or overwrite an existing one with the given UTF-8 "
        "content. The parent directory must already exist. Use `edit` to change "
        "part of an existing file instead of replacing it whole."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write.",
            },
            "content": {
                "type": "string",
                "description": "The full file content to write.",
            },
        },
        "required": ["path", "content"],
    },
)


@TOOLS.register("write")
def _build_write() -> Tool:
    return WriteTool()


class WriteTool(Tool):
    """Creates or overwrites a text file."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        path = str(arguments["path"])
        content = str(arguments["content"])
        try:
            written = write_text(path, content)
        except FileToolError as exc:
            return f"Error: {exc}"
        lines = content.count("\n") + 1 if content else 0
        return f"Wrote {written} bytes ({lines} lines) to {path}."

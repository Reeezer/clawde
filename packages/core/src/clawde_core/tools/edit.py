"""The ``edit`` tool — exact string replacement with a uniqueness guard.

The roadmap's flagged "deceptively hard one". The apply logic is deliberately
strict: an exact match that must be *unique* unless ``replace_all`` is set, so the
model never silently changes the wrong occurrence. We ship exact + unique-guard
only; the whitespace-tolerant "fuzzy" fallback is intentionally left out — it can
silently corrupt files, so it belongs in its own well-tested follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping

from clawde_core.models import ToolSpec
from clawde_core.tools._files import FileToolError, read_text, write_text
from clawde_core.tools.base import Tool
from clawde_core.tools.registry import TOOLS

_SPEC = ToolSpec(
    name="edit",
    description=(
        "Replace an exact substring in a file. `old_string` must occur exactly "
        "once unless `replace_all` is true; if it occurs more than once, add "
        "surrounding context to make it unique rather than guessing. Use `write` "
        "to create a file or replace it whole."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace (include enough context to be unique).",
            },
            "new_string": {
                "type": "string",
                "description": "Text to replace it with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of requiring a unique match.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)


@TOOLS.register("edit")
def _build_edit() -> Tool:
    return EditTool()


class EditTool(Tool):
    """Applies an exact string replacement to a file."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        path = str(arguments["path"])
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))

        if old == "":
            return "Error: 'old_string' is empty; use the `write` tool to create a file."
        if old == new:
            return "Error: 'old_string' and 'new_string' are identical; nothing to change."
        try:
            text = read_text(path)
        except FileToolError as exc:
            return f"Error: {exc}"

        count = text.count(old)
        if count == 0:
            return f"Error: 'old_string' not found in {path}."
        if count > 1 and not replace_all:
            return (
                f"Error: 'old_string' occurs {count} times in {path}. Add surrounding "
                "context to make it unique, or set replace_all=true to replace every match."
            )

        updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        try:
            write_text(path, updated)
        except FileToolError as exc:
            return f"Error: {exc}"
        if replace_all:
            return f"Edited {path}: replaced {count} occurrences."
        line = text[: text.index(old)].count("\n") + 1
        return f"Edited {path}: replaced 1 occurrence at line {line}."

"""The ``grep`` tool — search file contents with a regular expression.

Pure-Python ``re`` over globbed files: portable and offline-testable (a native
``ripgrep`` backend is a possible later optimisation). Binary, oversized, and
unreadable files are skipped rather than erroring, so one bad file never sinks a
whole search.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from clawde_core.models import ToolSpec
from clawde_core.tools._files import FileToolError, read_text
from clawde_core.tools.base import Tool
from clawde_core.tools.registry import TOOLS

_MAX_MATCHES = 200
_DEFAULT_INCLUDE = "**/*"

_SPEC = ToolSpec(
    name="grep",
    description=(
        "Search file contents with a regular expression and return matching lines "
        "as `path:line:text`. Limit which files are searched with `include` (a "
        "glob, default all files) under a base `path`. Use `glob` to find files by "
        "name instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search (default: current directory).",
            },
            "include": {
                "type": "string",
                "description": "Glob limiting which files are searched (default: all files).",
            },
        },
        "required": ["pattern"],
    },
)


@TOOLS.register("grep")
def _build_grep() -> Tool:
    return GrepTool()


class GrepTool(Tool):
    """Searches file contents with a regex."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        pattern = str(arguments["pattern"])
        base = Path(str(arguments.get("path", ".")))
        include = str(arguments.get("include", _DEFAULT_INCLUDE))
        if not base.is_dir():
            return f"Error: base directory not found: {base}"
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regular expression '{pattern}': {exc}"
        try:
            files = sorted(p for p in base.glob(include) if p.is_file())
        except ValueError as exc:
            return f"Error: invalid include pattern '{include}': {exc}"
        return _search(regex, files)


def _search(regex: re.Pattern[str], files: list[Path]) -> str:
    matches: list[str] = []
    for path in files:
        try:
            text = read_text(str(path))
        except FileToolError:
            continue  # skip binary / oversized / unreadable files
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{path.as_posix()}:{lineno}:{line}")
                if len(matches) >= _MAX_MATCHES:
                    matches.append(f"... (stopped at {_MAX_MATCHES} matches)")
                    return "\n".join(matches)
    if not matches:
        return f"No matches for '{regex.pattern}'."
    return "\n".join(matches)

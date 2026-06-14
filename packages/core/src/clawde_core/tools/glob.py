"""The ``glob`` tool — list files matching a glob pattern.

Finds files by name/path (e.g. ``**/*.py``); use ``grep`` to search file
contents. Backed by :meth:`pathlib.Path.glob`, which understands ``**`` for
recursive matching. Directories are filtered out — the agent wants files.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from clawde_core.models import ToolSpec
from clawde_core.tools.base import Tool
from clawde_core.tools.registry import TOOLS

_MAX_RESULTS = 1000

_SPEC = ToolSpec(
    name="glob",
    description=(
        "Find files matching a glob pattern (e.g. `**/*.py`) under a base "
        "directory. Returns matching file paths, one per line, sorted. Use this "
        "to locate files by name; use `grep` to search file contents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. `**/*.py` or `src/*.toml`.",
            },
            "path": {
                "type": "string",
                "description": "Base directory to search in (default: current directory).",
            },
        },
        "required": ["pattern"],
    },
)


@TOOLS.register("glob")
def _build_glob() -> Tool:
    return GlobTool()


class GlobTool(Tool):
    """Lists files matching a glob pattern."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        pattern = str(arguments["pattern"])
        base = Path(str(arguments.get("path", ".")))
        if not base.is_dir():
            return f"Error: base directory not found: {base}"
        try:
            matches = sorted(p for p in base.glob(pattern) if p.is_file())
        except ValueError as exc:
            return f"Error: invalid glob pattern '{pattern}': {exc}"
        if not matches:
            return f"No files match '{pattern}' under {base}."
        lines = [p.as_posix() for p in matches[:_MAX_RESULTS]]
        if len(matches) > _MAX_RESULTS:
            lines.append(f"... ({len(matches) - _MAX_RESULTS} more not shown)")
        return "\n".join(lines)

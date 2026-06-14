"""The ``bash`` tool — run a shell command and return its combined output.

Prefers a real ``bash`` (so commands stay portable, e.g. Git Bash on Windows),
falling back to the platform's default shell. Unsandboxed by design: permission
gating and sandboxing arrive in Phase 5 (see ``docs/ROADMAP.md``).
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping

from clawde_core.models import ToolSpec
from clawde_core.tools.base import Tool

_TIMEOUT_SECONDS = 120

_SPEC = ToolSpec(
    name="bash",
    description=(
        "Run a shell command and return its combined stdout and stderr. Use this "
        "to inspect the filesystem, run programs, and answer questions about the "
        "project. A non-zero exit status is reported in the output."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    },
)


class BashTool(Tool):
    """Executes shell commands on the host."""

    @property
    def spec(self) -> ToolSpec:
        return _SPEC

    def run(self, arguments: Mapping[str, object]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string."
        bash = shutil.which("bash")
        try:
            if bash is not None:
                completed = subprocess.run(
                    [bash, "-c", command],
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                )
            else:
                completed = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=_TIMEOUT_SECONDS,
                )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {_TIMEOUT_SECONDS}s."
        return _format_result(completed)


def _format_result(completed: subprocess.CompletedProcess[str]) -> str:
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode == 0:
        return output or "(no output)"
    return f"[exit {completed.returncode}] {output}".strip()

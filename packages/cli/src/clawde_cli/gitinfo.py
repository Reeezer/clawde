"""Read the current git branch and worktree for the status header (#12).

A best-effort, read-only peek at the repo so a session can announce where it is
working — most useful when several clawde sessions run in parallel, one per
``feature/*`` worktree. The git invocation is injected so the header is testable
without a real repo, and every failure degrades to ``None`` rather than raising:
the header simply omits what it can't learn.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

type GitRunner = Callable[[list[str]], str | None]

_GIT_TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class GitContext:
    """Where the session is working: the branch and the worktree folder name."""

    branch: str | None
    worktree: str | None


def current_git_context(run: GitRunner | None = None) -> GitContext:
    """The active branch and worktree folder, each ``None`` when unavailable."""
    git = run if run is not None else _run_git
    branch = git(["rev-parse", "--abbrev-ref", "HEAD"])
    top_level = git(["rev-parse", "--show-toplevel"])
    worktree = top_level.rsplit("/", 1)[-1] if top_level else None
    return GitContext(branch=branch, worktree=worktree)


def _run_git(args: list[str]) -> str | None:
    """Run ``git <args>`` and return its trimmed stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],  # fixed argv, no shell; git resolved from PATH by design
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None

"""The session status header (#12): which session this is, and what it's doing.

Distinct from the per-turn spinner (#9): this is the session's identity — its
branch/worktree, a short task title, and the active model and reasoning effort —
shown when the REPL starts so parallel sessions stay tellable apart. The line is
built from a typed :class:`HeaderState` by a pure function, so it renders the
same on screen and under test.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text

from clawde_cli.gitinfo import GitContext
from clawde_cli.session import Session

HEADER_STYLE = "#8fb5d6"  # calm pastel blue, distinct from the reply accents
_SEPARATOR = "  ·  "
_NO_TITLE = "(new session)"
_BRANCH_GLYPH = "⎇"


@dataclass(frozen=True)
class HeaderState:
    """Everything the status header shows, gathered into one immutable value."""

    branch: str | None
    worktree: str | None
    title: str | None
    provider: str
    model: str
    effort: str


def build_header_state(session: Session, git: GitContext) -> HeaderState:
    """Assemble the header's state from the live session and git context."""
    return HeaderState(
        branch=git.branch,
        worktree=git.worktree,
        title=session.title,
        provider=session.provider_name,
        model=session.model,
        effort=session.effort.value,
    )


def header_text(state: HeaderState) -> str:
    """The header as a single plain line (location · title · model · effort)."""
    parts = [
        _location(state),
        state.title or _NO_TITLE,
        f"{state.provider}/{state.model}",
        f"effort {state.effort}",
    ]
    return _SEPARATOR.join(parts)


def render_header(state: HeaderState) -> Text:
    """The header as a styled :class:`~rich.text.Text` for the REPL banner."""
    return Text(header_text(state), style=HEADER_STYLE)


def _location(state: HeaderState) -> str:
    if state.branch is None and state.worktree is None:
        return "no git"
    branch = state.branch or "detached"
    if state.worktree is not None:
        return f"{_BRANCH_GLYPH} {branch} @ {state.worktree}"
    return f"{_BRANCH_GLYPH} {branch}"

"""Signal when a turn finishes (#13) — so a backgrounded session is noticed.

The main pain with several clawde sessions running at once is missing which one
is done and waiting on you. After each REPL turn, ring the terminal bell (an
audible nudge for a session you're not watching) and print a bold, colour-coded
marker: the turn finished, failed, or was stopped, and clawde is awaiting input
again. The bell is silenceable (``CLAWDE_NO_BELL``); the marker is quiet enough
to always show. The bell only sounds on a real terminal — rich drops the control
code when output is redirected.
"""

from __future__ import annotations

import os

from rich.console import Console
from rich.text import Text

from clawde_cli.session import TurnOutcome, TurnStatus

_BELL_OFF_ENV = "CLAWDE_NO_BELL"

# Per outcome: the marker text and its style. Distinct treatment for a turn that
# finished, errored, or was interrupted (#13).
_MARKERS: dict[TurnStatus, tuple[str, str]] = {
    TurnStatus.OK: ("✓ ready", "bold green"),
    TurnStatus.ERROR: ("✗ failed", "bold red"),
    TurnStatus.INTERRUPTED: ("■ stopped", "bold yellow"),
}


def bell_enabled() -> bool:
    """Whether to ring the bell on completion (set ``CLAWDE_NO_BELL`` to silence)."""
    return os.environ.get(_BELL_OFF_ENV) is None


def signal_done(console: Console, outcome: TurnOutcome, *, bell: bool = True) -> None:
    """Announce a finished turn: an optional bell, then a colour-coded marker."""
    if bell:
        console.bell()
    label, style = _MARKERS[outcome.status]
    console.print(Text(label, style=style))

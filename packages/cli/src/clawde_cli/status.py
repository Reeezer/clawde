"""The working-status line: a spinner, a whimsical verb, elapsed time, tokens.

While the agent thinks or runs a tool, clawde shows a single live line like
``✻ Wandering… (8m 36s · ↓ 46.3k tokens)`` to signal progress. This module owns
that line's *content*; :mod:`clawde_cli.rendering` owns *when* it is shown (one
shared live region — it never competes with the streamed markdown).

Everything time- and chance-dependent is injected — the clock and the verb
chooser — so the line renders deterministically under test with no sleeping and
no real RNG (the glyph renders only to a real terminal, never a redirected pipe).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence

from clawde_core.models import Usage
from rich.text import Text

SPINNER_GLYPH = "✻"
RECEIVED_GLYPH = "↓"  # tokens received from the model (output tokens)
SPINNER_STYLE = "#c3a6ff"  # pastel purple, matching the inline-code accent
VERB_STYLE = "bold"
DETAIL_STYLE = "dim"
SUMMARY_STYLE = "#f5e0a3"  # pastel yellow, for the turn's closing summary
REFRESH_PER_SECOND = 8  # spinner repaints/sec while the agent works

# A curated, deliberately whimsical pool — the verb rotates each working phase.
DEFAULT_VERBS = (
    "Wandering",
    "Pondering",
    "Conjuring",
    "Tinkering",
    "Noodling",
    "Ruminating",
    "Percolating",
    "Scheming",
    "Puzzling",
    "Spelunking",
)

_SECONDS_PER_MINUTE = 60
_TOKENS_PER_K = 1000


def format_elapsed(seconds: float) -> str:
    """Format a duration as ``8m 36s`` (or ``36s`` under a minute)."""
    whole = int(seconds)
    minutes, secs = divmod(whole, _SECONDS_PER_MINUTE)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def format_token_count(tokens: int) -> str:
    """Format a token count compactly: ``512`` below 1k, else ``46.3k``."""
    if tokens < _TOKENS_PER_K:
        return str(tokens)
    return f"{tokens / _TOKENS_PER_K:.1f}k"


def format_summary(elapsed: float, tokens: int) -> Text:
    """A turn's closing line — wall-clock time and tokens used, in pastel yellow."""
    summary = f"Done in {format_elapsed(elapsed)} · {format_token_count(tokens)} tokens"
    return Text(summary, style=SUMMARY_STYLE)


def _random_verb(verbs: Sequence[str]) -> str:
    return random.choice(verbs)


class StatusReporter:
    """The live status line's state: when work began, the verb, and tokens so far.

    Renders itself (``__rich__``) to the spinner line. The clock and verb chooser
    are injected so the line is deterministic under test.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        verbs: Sequence[str] = DEFAULT_VERBS,
        choose: Callable[[Sequence[str]], str] = _random_verb,
    ) -> None:
        self._clock = clock
        self._verbs = tuple(verbs)
        self._choose = choose
        self._start: float | None = None
        self._verb = self._verbs[0]
        self._usage = Usage()

    def start(self) -> None:
        """Mark the turn's start — the moment elapsed time counts from."""
        self._start = self._clock()

    def next_verb(self) -> None:
        """Rotate to a fresh verb for a new working phase."""
        self._verb = self._choose(self._verbs)

    def update(self, usage: Usage) -> None:
        """Record the running token usage to display."""
        self._usage = usage

    def elapsed(self) -> float:
        """Seconds since :meth:`start` — 0 before the turn begins."""
        return 0.0 if self._start is None else self._clock() - self._start

    def __rich__(self) -> Text:
        line = Text()
        line.append(f"{SPINNER_GLYPH} ", style=SPINNER_STYLE)
        line.append(f"{self._verb}… ", style=VERB_STYLE)
        detail = (
            f"({format_elapsed(self.elapsed())} · "
            f"{RECEIVED_GLYPH} {format_token_count(self._usage.output_tokens)} tokens)"
        )
        line.append(detail, style=DETAIL_STYLE)
        return line

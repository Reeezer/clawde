"""The working-status line: a spinner, a whimsical verb, elapsed time, tokens.

While the agent thinks or runs a tool, clawde shows a single live line like
``✻ Wandering… (8m 36s · ↑ 12.0k ↓ 1.5k tokens)`` to signal progress: ``↑`` is the
tokens sent to the model so far this turn, ``↓`` the tokens received. This module owns
that line's *content*; :mod:`clawde_cli.rendering` owns *when* it is shown (one
shared live region — it never competes with the streamed markdown).

The leading glyph *pulses*: it cycles through :data:`SPINNER_FRAMES` as a pure
function of elapsed time, so it animates on a real terminal with no mutable frame
counter — and stays deterministic under test. The clock and the verb chooser are
injected, so the line renders with no sleeping and no real RNG.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence

from clawde_core.models import TokenBudget, Usage
from rich.text import Text

SPINNER_FRAMES = ("✢", "✻", "✽", "✻")  # pulse: small → large → small (no emoji)
SENT_GLYPH = "↑"  # tokens sent to the model so far this turn (input tokens)
RECEIVED_GLYPH = "↓"  # tokens received from the model so far this turn (output tokens)
CONTEXT_LABEL = "ctx"  # the running context read, shown as `ctx used/limit` (e.g. ctx 24.1k/1M)
CONTEXT_PENDING = "–"  # placeholder for `used` before the first reply is measured (ctx –/1M)
SPINNER_STYLE = "#f5be8c"  # pastel orange — soft, like the other reply accents
VERB_STYLE = "bold"
DETAIL_STYLE = "dim"
SUMMARY_STYLE = "#f5e0a3"  # pastel yellow, for the turn's closing summary
REFRESH_PER_SECOND = 8  # spinner repaints/sec while the agent works
SPINNER_FRAME_SECONDS = 0.7  # how long each frame is held; raise to slow the pulse

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
_TOKENS_PER_M = 1_000_000


def format_elapsed(seconds: float) -> str:
    """Format a duration as ``8m 36s`` (or ``36s`` under a minute)."""
    whole = int(seconds)
    minutes, secs = divmod(whole, _SECONDS_PER_MINUTE)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def spinner_frame(elapsed: float) -> str:
    """The pulse glyph for an elapsed time — a pure function of the clock (no
    mutable state), so the star animates on a real terminal yet stays
    deterministic under test. Holds each :data:`SPINNER_FRAMES` entry for
    :data:`SPINNER_FRAME_SECONDS`.
    """
    return SPINNER_FRAMES[int(elapsed / SPINNER_FRAME_SECONDS) % len(SPINNER_FRAMES)]


def format_token_count(tokens: int) -> str:
    """Format a token count compactly: ``512`` below 1k, ``46.3k`` below 1M, else ``1.0M``."""
    if tokens < _TOKENS_PER_K:
        return str(tokens)
    if tokens < _TOKENS_PER_M:
        return f"{tokens / _TOKENS_PER_K:.1f}k"
    return f"{tokens / _TOKENS_PER_M:.1f}M"


def format_context(budget: TokenBudget | None, window: int | None = None) -> str | None:
    """The running context read as ``ctx used/limit`` — e.g. ``ctx 24.1k/1M``.

    Before the first reply is measured, ``budget`` is ``None`` but the model's
    ``window`` is already known, so a ``–`` placeholder stands in for ``used``
    (``ctx –/1M``). Returns ``None`` when neither is known yet.

    ``used`` keeps its live precision (``24.1k``); the window is shown as a round
    figure (``1M``, ``200k``) since it does not change mid-turn.
    """
    if budget is not None:
        return f"{CONTEXT_LABEL} {format_token_count(budget.used)}/{_format_limit(budget.limit)}"
    if window is not None:
        return f"{CONTEXT_LABEL} {CONTEXT_PENDING}/{_format_limit(window)}"
    return None


def _format_limit(limit: int) -> str:
    return format_token_count(limit).replace(".0", "")


def _token_detail(usage: Usage, budget: TokenBudget | None, window: int | None) -> list[str]:
    """The shared tail of the spinner and summary: an optional ``ctx`` read, then
    the cumulative ``↑`` sent / ``↓`` received totals."""
    parts: list[str] = []
    context = format_context(budget, window)
    if context is not None:
        parts.append(context)
    parts.append(
        f"{SENT_GLYPH} {format_token_count(usage.input_tokens)} "
        f"{RECEIVED_GLYPH} {format_token_count(usage.output_tokens)} tokens"
    )
    return parts


def format_summary(elapsed: float, usage: Usage, budget: TokenBudget | None = None) -> Text:
    """A turn's closing line — time, the context read, and tokens sent/received.

    Uses the same ``ctx used/limit`` context read and ``↑`` sent / ``↓`` received
    glyphs as the live spinner; ``ctx`` is shown only when a ``budget`` is given.
    """
    detail = " · ".join([f"Done in {format_elapsed(elapsed)}", *_token_detail(usage, budget, None)])
    return Text(detail, style=SUMMARY_STYLE)


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
        self._budget: TokenBudget | None = None
        self._context_window: int | None = None

    def start(self) -> None:
        """Mark the turn's start — the moment elapsed time counts from."""
        self._start = self._clock()

    def next_verb(self) -> None:
        """Rotate to a fresh verb for a new working phase."""
        self._verb = self._choose(self._verbs)

    def update(self, usage: Usage) -> None:
        """Record the running token usage (sent + received) to display."""
        self._usage = usage

    def set_budget(self, budget: TokenBudget) -> None:
        """Record the running context read, shown as ``ctx used/limit``."""
        self._budget = budget

    def set_context_window(self, limit: int) -> None:
        """Record the model's context window — shown as ``ctx –/limit`` until the
        first reply reports real usage."""
        self._context_window = limit

    def elapsed(self) -> float:
        """Seconds since :meth:`start` — 0 before the turn begins."""
        return 0.0 if self._start is None else self._clock() - self._start

    def __rich__(self) -> Text:
        elapsed = self.elapsed()  # read the clock once so the glyph and the time agree
        line = Text()
        line.append(f"{spinner_frame(elapsed)} ", style=SPINNER_STYLE)
        line.append(f"{self._verb}… ", style=VERB_STYLE)
        detail = " · ".join(
            [
                format_elapsed(elapsed),
                *_token_detail(self._usage, self._budget, self._context_window),
            ]
        )
        line.append(f"({detail})", style=DETAIL_STYLE)
        return line

"""The no-op compactor — compaction disabled (``strategy = "off"``).

Returns the conversation untouched, so the loop never summarises. The trivial
second implementation that makes compaction a genuinely swappable stage and lets
a session opt out of mid-loop model calls (ADR-0005).
"""

from __future__ import annotations

from clawde_core.context.compaction import COMPACTORS
from clawde_core.context.compaction.base import Compactor
from clawde_core.context.history import History
from clawde_core.providers.base import ModelProvider


class NoOpCompactor(Compactor):
    """Leaves the conversation unchanged — compaction turned off."""

    def compact(self, history: History, *, provider: ModelProvider) -> History:
        return history


@COMPACTORS.register("off")
def _build_off() -> Compactor:
    """Build the no-op compactor — the ``COMPACTORS`` builder for ``off``."""
    return NoOpCompactor()

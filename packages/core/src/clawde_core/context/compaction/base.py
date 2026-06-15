"""The ``Compactor`` ABC — clawde's contract for shrinking a long conversation.

When a session nears the model's context window (ADR-0004), the loop asks its
compactor to rewrite the :class:`~clawde_core.context.history.History` into a
shorter one — typically by summarising old turns and keeping the recent ones.
Strategies are swappable behind this ABC in the ``COMPACTORS`` registry, resolved
from :class:`~clawde_core.config.Settings` by the factory (ADR-0003 / ADR-0005);
the loop depends only on this contract, never on a concrete strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from clawde_core.context.history import History
from clawde_core.providers.base import ModelProvider


class Compactor(ABC):
    """Rewrites a conversation to fit the window while preserving the thread."""

    @abstractmethod
    def compact(self, history: History, *, provider: ModelProvider) -> History:
        """Return a new — ideally shorter — :class:`History` for the conversation.

        ``provider`` is offered for strategies that summarise by calling the model.
        The result must keep whole turns (never split a tool result from the call
        it answers) so the conversation stays valid to send. A strategy that cannot
        shrink the history returns *the same* ``History`` object; the loop detects
        that no-op by identity and won't ask again this turn (it compacts at most
        once per turn — ADR-0005).
        """

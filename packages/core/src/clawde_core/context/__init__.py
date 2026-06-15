"""Context management (Phase 4): the conversation's home and its token budget.

The :class:`~clawde_core.context.history.History` store owns the message list the
loop used to hold inline; token budgeting rides on the provider contract
(``context_window`` / ``count_tokens``, ADR-0004) and surfaces as a
:class:`~clawde_core.models.TokenBudget`. Compaction (summarising old turns)
lands here next and acts on that budget.
"""

from __future__ import annotations

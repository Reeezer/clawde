"""Conversation compaction (Phase 4): swappable strategies for shrinking history.

When a session nears the model's context window, the loop asks a
:class:`~clawde_core.context.compaction.base.Compactor` to rewrite its
:class:`~clawde_core.context.history.History` into a shorter one. Strategies
self-register a settings-reading builder with the ``COMPACTORS`` registry; the
:func:`~clawde_core.context.compaction.factory.build_compactor` factory resolves
the one named in :class:`~clawde_core.config.Settings` (ADR-0003 / ADR-0005).
"""

from __future__ import annotations

from clawde_core.context.compaction.base import Compactor
from clawde_core.registry import Registry

COMPACTORS: Registry[Compactor] = Registry("compactors")

"""The compactor factory — resolves ``Settings`` to a built ``Compactor``.

Reads the configured strategy (``Settings.compaction.strategy``) and builds it
from the ``COMPACTORS`` registry. Importing the impls runs their
``@COMPACTORS.register(...)`` side effect (each is import-time light, no model
SDK). The loop and CLI depend only on this function, never on a concrete strategy
(ADR-0003 / ADR-0005).
"""

from __future__ import annotations

from clawde_core.config import Settings, get_settings
from clawde_core.context.compaction import COMPACTORS
from clawde_core.context.compaction import off as _off  # noqa: F401
from clawde_core.context.compaction import summarise as _summarise  # noqa: F401
from clawde_core.context.compaction.base import Compactor


def build_compactor(settings: Settings | None = None) -> Compactor:
    """Build the configured compaction strategy (default ``summarise``)."""
    settings = settings if settings is not None else get_settings()
    return COMPACTORS.create(settings.compaction.strategy)

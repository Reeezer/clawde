"""clawde's model providers (BYOM).

Each provider is one file implementing the
:class:`~clawde_core.providers.base.ModelProvider` ABC, normalising a backend's
wire format to and from clawde's typed models. Concrete providers self-register a
zero-arg, settings-reading builder with the ``PROVIDERS`` registry; the
:func:`~clawde_core.providers.factory.build_provider` factory resolves the one
named in :class:`~clawde_core.config.Settings` (see ADR-0003).
"""

from __future__ import annotations

from clawde_core.providers.base import ModelProvider
from clawde_core.registry import Registry

PROVIDERS: Registry[ModelProvider] = Registry("providers")

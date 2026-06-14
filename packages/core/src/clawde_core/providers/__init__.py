"""clawde's model providers (BYOM).

Each provider is one file implementing the
:class:`~clawde_core.providers.base.ModelProvider` ABC, normalising a backend's
wire format to and from clawde's typed models. The ``PROVIDERS`` registry and
the ``Settings`` factory arrive in Phase 2 (see ADR-0003); Phase 1 ships only
``gemini``.
"""

from __future__ import annotations

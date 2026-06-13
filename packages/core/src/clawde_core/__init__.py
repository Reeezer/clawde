"""clawde_core — the bring-your-own-model agent engine.

Houses the agent loop, the provider abstraction (BYOM), the tool system, and
context management. Everything swappable lives behind an ABC + a named
:class:`Registry`; see ``CLAUDE.md`` and ``docs/adr/`` for the architecture.
"""

from __future__ import annotations

from clawde_core.config import Settings, get_settings
from clawde_core.registry import Registry, RegistryError

__version__ = "0.0.1"

__all__ = [
    "Registry",
    "RegistryError",
    "Settings",
    "__version__",
    "get_settings",
]

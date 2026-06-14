"""clawde's tools: capabilities the agent can invoke.

Each tool is one file implementing the :class:`~clawde_core.tools.base.Tool`
ABC. The ``TOOLS`` registry and the full tool set (read / write / edit / glob /
grep) arrive in Phase 3; Phase 1 ships only ``bash``.
"""

from __future__ import annotations

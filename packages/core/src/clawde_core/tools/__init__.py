"""clawde's tools: capabilities the agent can invoke.

Each tool is one file implementing the :class:`~clawde_core.tools.base.Tool` ABC
and registering a builder with the ``TOOLS`` registry. Importing this package
imports every tool module below, so the ``@TOOLS.register`` decorators run and the
registry is populated; the loop and CLI then build their tool set from the factory
in :mod:`clawde_core.tools.registry` rather than importing concrete tools.
"""

from __future__ import annotations

from clawde_core.tools import bash, edit, glob, grep, read, write

__all__ = ["bash", "edit", "glob", "grep", "read", "write"]

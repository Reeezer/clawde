"""The ``TOOLS`` registry and the factory that builds the agent's tool set.

The tool side of clawde's swappable architecture (ADR-0003), mirroring the
planned ``PROVIDERS`` side: each tool registers a zero-argument builder under its
name with ``@TOOLS.register("name")``, and the loop/CLI ask the factory for tools
rather than importing concrete classes. Adding a tool is "write the file and
register it" — nothing here changes.

Registration is an import side effect: importing the ``clawde_core.tools``
package imports every tool module (see ``tools/__init__.py``), running each
decorator. This module therefore stays free of concrete-tool imports, keeping the
dependency graph acyclic (tool modules import this, not the other way round).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from clawde_core.registry import Registry
from clawde_core.tools.base import Tool

if TYPE_CHECKING:
    from clawde_core.config import Settings

TOOLS: Registry[Tool] = Registry("tools")


def create_tool(name: str) -> Tool:
    """Build the tool registered under ``name`` (``RegistryError`` if unknown)."""
    return TOOLS.create(name)


def build_tools(settings: Settings) -> list[Tool]:
    """Build the agent's active tool set, ordered by name.

    Today every registered tool is active; ``settings`` is the seam through which
    later phases (permission profiles, opt-outs) will narrow the set without the
    loop or CLI changing.
    """
    # Registry.keys() returns a sorted tuple (not a dict view), so .keys() is required.
    return [TOOLS.create(name) for name in TOOLS.keys()]  # noqa: SIM118

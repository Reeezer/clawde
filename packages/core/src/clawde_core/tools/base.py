"""The ``Tool`` ABC — clawde's contract for a capability the model can invoke.

One tool per file under ``tools/`` (see ``CLAUDE.md``). A tool advertises itself
to the model with a :class:`~clawde_core.models.ToolSpec` (name, description,
JSON-Schema parameters) and executes a call via :meth:`Tool.run`. The full
``TOOLS`` registry and JSON-Schema dispatch land in Phase 3; for now the loop
holds tools directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from clawde_core.models import ToolSpec


class Tool(ABC):
    """A capability the agent can call. Concrete tools live one per file."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """How this tool advertises itself to the model."""

    @abstractmethod
    def run(self, arguments: Mapping[str, object]) -> str:
        """Execute the tool with ``arguments`` and return its output as text."""

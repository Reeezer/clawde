"""The ``Tool`` ABC — clawde's contract for a capability the model can invoke.

One tool per file under ``tools/`` (see ``CLAUDE.md``). A tool advertises itself
to the model with a :class:`~clawde_core.models.ToolSpec` (name, description,
JSON-Schema parameters) and implements the abstract :meth:`Tool.run`. The loop
never calls ``run`` directly: it goes through :meth:`Tool.invoke`, which validates
the model's arguments against the tool's own schema *before* dispatch. A bad call
(missing field, wrong type) becomes a recoverable error string the model can read
and retry — it never raises across the loop boundary — so each tool's ``run`` can
trust its arguments instead of re-checking them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import jsonschema
from jsonschema.exceptions import best_match

from clawde_core.models import ToolSpec


class Tool(ABC):
    """A capability the agent can call. Concrete tools live one per file."""

    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """How this tool advertises itself to the model."""

    @abstractmethod
    def run(self, arguments: Mapping[str, object]) -> str:
        """Execute the tool with validated ``arguments`` and return its output.

        Called only by :meth:`invoke`, so ``arguments`` already satisfy
        ``spec.parameters`` — implementations need not re-check required fields
        or argument types.
        """

    def invoke(self, arguments: Mapping[str, object]) -> str:
        """Validate ``arguments`` against ``spec.parameters``, then run the tool.

        Returns a model-readable ``Error: …`` string if the arguments violate
        the schema; otherwise delegates to :meth:`run`. Never raises for bad
        arguments — the loop feeds the error back so the model can correct it.
        """
        error = _schema_error(self.spec.parameters, arguments)
        if error is not None:
            return f"Error: {error}"
        return self.run(arguments)


def _schema_error(schema: Mapping[str, object], arguments: Mapping[str, object]) -> str | None:
    """Return the most relevant JSON-Schema violation as text, or ``None`` if valid."""
    validator = jsonschema.Draft202012Validator(dict(schema))
    error = best_match(validator.iter_errors(dict(arguments)))
    return None if error is None else error.message

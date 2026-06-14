"""The ``ModelProvider`` ABC — clawde's contract for a model backend (BYOM).

Concrete providers (Anthropic, OpenAI, Gemini, Ollama, …) live one per file and
normalise their wire format to and from clawde's typed
:mod:`~clawde_core.models`. The loop depends only on this ABC and never sees a
vendor type (ADR-0003). The ``PROVIDERS`` registry, the ``Settings`` factory,
streaming, and the JSON-in-text tool-calling fallback arrive in Phase 2; the
contract below is what the loop is written against and will not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from clawde_core.models import Completion, Message, ToolSpec


class ProviderError(RuntimeError):
    """A model backend could not be reached or returned an unusable response."""


class ModelProvider(ABC):
    """Adapts one model backend to clawde's typed message / tool / usage models."""

    @abstractmethod
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        """Send the conversation and available tools; return the model's reply.

        Implementations translate ``messages`` and ``tools`` to their wire
        format, call the backend, and normalise the result back into a
        :class:`~clawde_core.models.Completion`.
        """

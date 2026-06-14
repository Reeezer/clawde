"""The ``ModelProvider`` ABC — clawde's contract for a model backend (BYOM).

Concrete providers (Anthropic, OpenAI, Gemini, Ollama, …) live one per file and
normalise their wire format to and from clawde's typed
:mod:`~clawde_core.models`. The loop depends only on this ABC and never sees a
vendor type (ADR-0003). The ``PROVIDERS`` registry, the ``Settings`` factory,
and the JSON-in-text tool-calling fallback arrive in Phase 2; the contract below
is what the loop is written against and will not change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

from clawde_core.models import Completion, Message, StreamChunk, ToolSpec


class ProviderError(RuntimeError):
    """A model backend could not be reached or returned an unusable response."""


class ModelProvider(ABC):
    """Adapts one model backend to clawde's typed message / tool / usage models.

    Image contract: clawde never drops input silently. A backend that accepts
    images translates :class:`~clawde_core.models.ImageContent` to its wire
    format; one that does not must call :meth:`_reject_image_input` before
    translating a conversation, so an attached image raises a clear
    :class:`ProviderError` instead of vanishing.
    """

    @abstractmethod
    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        """Send the conversation and available tools; return the model's reply.

        Implementations translate ``messages`` and ``tools`` to their wire
        format, call the backend, and normalise the result back into a
        :class:`~clawde_core.models.Completion`.
        """

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        """Stream the reply as text deltas, ending with a chunk that carries the
        full :class:`~clawde_core.models.Completion`.

        Default: no real streaming — emit the whole reply as one terminal chunk.
        Providers with a streaming API (e.g. Gemini) override this.
        """
        yield StreamChunk(completion=self.complete(messages, tools))

    def _reject_image_input(self, messages: Sequence[Message]) -> None:
        """Raise if any message carries images this backend can't send.

        Text-only providers call this first thing in :meth:`complete` /
        :meth:`stream`; image-capable providers (e.g. Gemini) handle the image
        parts themselves and skip it.
        """
        if any(message.images for message in messages):
            raise ProviderError(f"{type(self).__name__} does not support image input.")

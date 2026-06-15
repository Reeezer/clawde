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
from collections.abc import Iterator, Mapping, Sequence

from clawde_core.models import Completion, Message, ReasoningEffort, StreamChunk, ToolSpec


class ProviderError(RuntimeError):
    """A model backend could not be reached or returned an unusable response."""


DEFAULT_CONTEXT_WINDOW = 128_000
"""Conservative fallback window for a provider that doesn't know its model's."""

_CHARS_PER_TOKEN = 4


def _estimate_tokens(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
    """A cheap, offline token estimate: ~4 characters per token plus per-message
    framing. The :class:`ModelProvider` fallback when no exact counter exists
    (ADR-0004)."""
    text = "".join(message.content for message in messages)
    text += "".join(tool.name + tool.description + str(tool.parameters) for tool in tools)
    return len(text) // _CHARS_PER_TOKEN + len(messages)


class ModelProvider(ABC):
    """Adapts one model backend to clawde's typed message / tool / usage models.

    Image contract: clawde never drops input silently. A backend that accepts
    images translates :class:`~clawde_core.models.ImageContent` to its wire
    format; one that does not must call :meth:`_reject_image_input` before
    translating a conversation, so an attached image raises a clear
    :class:`ProviderError` instead of vanishing.

    Reasoning effort: :meth:`set_reasoning_effort` is the runtime toggle the
    factory seeds from ``Settings`` and the CLI's ``--effort`` (later ``/effort``)
    drives. Support is per *model*, not per provider, so each impl resolves the
    effort against a per-model mapping via :meth:`_resolve_effort`: an
    unsupported level raises rather than being silently clamped, and ``OFF`` is
    always honoured by sending no reasoning field.
    """

    _reasoning_effort: ReasoningEffort = ReasoningEffort.OFF

    @property
    def reasoning_effort(self) -> ReasoningEffort:
        """The reasoning effort applied to subsequent calls (default ``OFF``)."""
        return self._reasoning_effort

    def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
        """Set the reasoning effort for subsequent calls — the runtime toggle."""
        self._reasoning_effort = effort

    def _resolve_effort(
        self, supported: Mapping[ReasoningEffort, str], *, model: str
    ) -> str | None:
        """Translate the current effort to ``model``'s wire value, or ``None`` for OFF.

        ``supported`` maps each effort ``model`` honours to its backend value. A
        non-OFF effort outside it raises :class:`ProviderError` — clawde never
        clamps a level a model can't honour, it reports the mismatch up front.
        """
        effort = self._reasoning_effort
        if effort is ReasoningEffort.OFF:
            return None
        wire = supported.get(effort)
        if wire is None:
            raise ProviderError(_unsupported_effort(model, effort, supported))
        return wire

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

    @property
    def context_window(self) -> int:
        """The active model's maximum input tokens.

        Conservative fallback; a provider that knows its model's real window
        overrides this (ADR-0004).
        """
        return DEFAULT_CONTEXT_WINDOW

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        """Estimate the token cost of sending ``messages`` and ``tools``.

        Default: an offline ~4-chars-per-token heuristic. Real providers override
        with their exact tokenizer — Anthropic / Gemini count endpoints, or
        ``tiktoken`` for OpenAI-compatible (ADR-0004).
        """
        return _estimate_tokens(messages, tools)

    def _reject_image_input(self, messages: Sequence[Message]) -> None:
        """Raise if any message carries images this backend can't send.

        Text-only providers call this first thing in :meth:`complete` /
        :meth:`stream`; image-capable providers (e.g. Gemini) handle the image
        parts themselves and skip it.
        """
        if any(message.images for message in messages):
            raise ProviderError(f"{type(self).__name__} does not support image input.")


def _unsupported_effort(
    model: str, effort: ReasoningEffort, supported: Mapping[ReasoningEffort, str]
) -> str:
    """Compose the error for a reasoning effort ``model`` can't honour."""
    if not supported:
        return f"Model {model!r} does not support reasoning effort; use effort 'off' (the default)."
    levels = ", ".join(level.value for level in ReasoningEffort if level in supported)
    return (
        f"Model {model!r} does not support reasoning effort {effort.value!r}; "
        f"it supports: off, {levels}."
    )

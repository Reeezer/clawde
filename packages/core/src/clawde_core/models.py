"""Typed, immutable models that flow across clawde's stages.

These are the lingua franca of the agent: providers normalise their wire
formats to and from these, tools advertise themselves with a :class:`ToolSpec`
and return a :class:`ToolResult`, and the loop speaks only in :class:`Message`s.
Nothing crosses a stage boundary as an ad-hoc dict (see ``CLAUDE.md``).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base for clawde's value objects: validated, immutable, comparable."""

    model_config = ConfigDict(frozen=True)


class Role(StrEnum):
    """Who authored a :class:`Message`."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ReasoningEffort(StrEnum):
    """How hard the model should think before answering — normalised across backends.

    A single knob clawde maps to each provider's own control (Anthropic's
    ``output_config.effort`` paired with adaptive thinking, OpenAI's
    ``reasoning_effort``, Gemini's ``thinking_level``). ``OFF`` means clawde sends
    no reasoning control and the backend keeps its own default; the rest climb
    from a quick pass to maximum deliberation. Backends that top out lower clamp
    to their ceiling (Gemini has no ``xhigh`` / ``max``; OpenAI has no ``max``).
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ToolCall(_Frozen):
    """A model's request to run a tool with arguments.

    ``signature`` carries an opaque, provider-issued token bound to this call that
    some backends require sent back unchanged on the next turn — Gemini 3 signs
    each function call with a ``thought_signature`` and rejects a follow-up that
    drops it. The loop never inspects it; it rides along on the assistant turn so
    the replayed request stays valid, mirroring :class:`ThinkingBlock`. Backends
    that don't sign tool calls (Anthropic, OpenAI) leave it ``None``.
    """

    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)
    signature: bytes | None = None


class ToolResult(_Frozen):
    """The output of a tool execution, handed back to the model."""

    tool_call_id: str
    content: str


class Usage(_Frozen):
    """Token counts a provider reports for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class TokenBudget(_Frozen):
    """How much of the model's context window the conversation occupies.

    ``limit`` is the active model's window (``ModelProvider.context_window``);
    ``used`` is the input-token cost of the conversation right now. The signal
    Phase 4 compaction acts on (ADR-0004).
    """

    limit: int
    used: int

    @property
    def remaining(self) -> int:
        """Input tokens still free before the window is full (never negative)."""
        return max(self.limit - self.used, 0)

    @property
    def fraction(self) -> float:
        """Share of the window used — may exceed 1.0 when over budget."""
        if self.limit <= 0:
            return 1.0
        return self.used / self.limit


class ImageContent(_Frozen):
    """An image attached to a user message: raw bytes plus their media type.

    The typed alternative to smuggling base64 blobs through ``content``: a
    provider that accepts images translates these to its wire format, and one
    that doesn't rejects them rather than dropping them silently (see
    :class:`~clawde_core.providers.base.ModelProvider`).
    """

    mime_type: str
    data: bytes


class ThinkingBlock(_Frozen):
    """A block of the model's reasoning, preserved verbatim so it can be replayed.

    With extended thinking *and* tool use, a backend may require the reasoning
    that preceded a tool call to be sent back unchanged on the next turn or it
    rejects the request (Anthropic signs each block; a *redacted* block — Claude's
    reasoning was flagged — carries only opaque ``redacted_data`` and no text).
    The loop never inspects these; it carries them on the assistant turn that
    produced them so the next request stays valid.
    """

    text: str = ""
    signature: str | None = None
    redacted_data: str | None = None


class Message(_Frozen):
    """One entry in the conversation.

    A single shape covers every role; which fields are populated depends on
    ``role`` (user turns may carry ``images``; assistant turns carry
    ``tool_calls`` and any ``thinking`` blocks; tool results carry
    ``tool_call_id`` and ``name``). The constructors below make intent explicit.
    """

    role: Role
    content: str = ""
    images: tuple[ImageContent, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: tuple[ThinkingBlock, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, images: tuple[ImageContent, ...] = ()) -> Message:
        return cls(role=Role.USER, content=content, images=images)

    @classmethod
    def assistant(
        cls,
        content: str = "",
        tool_calls: tuple[ToolCall, ...] = (),
        thinking: tuple[ThinkingBlock, ...] = (),
    ) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls, thinking=thinking)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class Completion(_Frozen):
    """What a provider returns from one call: text and/or tool calls, plus usage.

    ``thinking`` carries any reasoning blocks the backend returned (extended
    thinking); the loop preserves them on the assistant turn so a follow-up
    request with tool results stays valid.
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: tuple[ThinkingBlock, ...] = ()
    usage: Usage = Field(default_factory=Usage)


class StreamChunk(_Frozen):
    """One piece of a streamed reply: an incremental text delta, plus — on the
    final chunk — the fully assembled :class:`Completion`.
    """

    text: str = ""
    completion: Completion | None = None


class ToolSpec(_Frozen):
    """A tool's advertisement to the model: its name, purpose, and JSON-Schema args."""

    name: str
    description: str
    parameters: dict[str, object] = Field(default_factory=dict)

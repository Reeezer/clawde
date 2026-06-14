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


class ToolCall(_Frozen):
    """A model's request to run a tool with arguments."""

    id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


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


class Message(_Frozen):
    """One entry in the conversation.

    A single shape covers every role; which fields are populated depends on
    ``role`` (assistant turns carry ``tool_calls``; tool results carry
    ``tool_call_id`` and ``name``). The constructors below make intent explicit.
    """

    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: tuple[ToolCall, ...] = ()) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class Completion(_Frozen):
    """What a provider returns from one call: text and/or tool calls, plus usage."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Field(default_factory=Usage)


class ToolSpec(_Frozen):
    """A tool's advertisement to the model: its name, purpose, and JSON-Schema args."""

    name: str
    description: str
    parameters: dict[str, object] = Field(default_factory=dict)

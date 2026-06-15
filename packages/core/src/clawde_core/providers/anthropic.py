"""The Anthropic provider — clawde talking to Claude via the ``anthropic`` SDK.

Lazy-imports ``anthropic`` inside the call (the optional ``anthropic`` extra,
ADR-0003) so the base install stays SDK-free. Translates clawde's typed messages
and tools into the Messages API wire shapes, calls ``messages.create`` (or
``messages.stream`` for SSE), and normalises the reply (text, tool use, token
usage) back into clawde's models. Claude is multimodal, so image content is
translated rather than rejected.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from clawde_core.config import get_settings
from clawde_core.models import Completion, Message, Role, StreamChunk, ToolCall, ToolSpec, Usage
from clawde_core.providers import PROVIDERS
from clawde_core.providers.base import ModelProvider, ProviderError

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import Usage as AnthropicUsage

DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_MAX_TOKENS = 8192
# All current Claude models share a 200k-token input window (ADR-0004).
_CONTEXT_WINDOW = 200_000

# untyped: the anthropic SDK's create()/stream() take TypedDict params; clawde
# builds the equivalent wire dicts and lets the SDK validate them at the boundary.
type _Wire = dict[str, Any]


class AnthropicProvider(ModelProvider):
    """Talks to the Anthropic Messages API via the ``anthropic`` SDK."""

    def __init__(
        self, *, api_key: str, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> None:
        if not api_key:
            raise ProviderError(
                "Anthropic API key is missing. "
                "Set CLAWDE_PROVIDERS__ANTHROPIC__API_KEY in your .env."
            )
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._client_cache: Anthropic | None = None

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        response = self._client().messages.create(**self._request(messages, tools))
        return _to_completion(response)

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        with self._client().messages.stream(**self._request(messages, tools)) as stream:
            for text in stream.text_stream:
                yield StreamChunk(text=text)
            final = stream.get_final_message()
        yield StreamChunk(completion=_to_completion(final))

    @property
    def context_window(self) -> int:
        return _CONTEXT_WINDOW

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        """Exact, server-side count via the Anthropic ``messages.count_tokens`` API.

        An on-demand call (ADR-0004): the hot loop budgets on reported usage, so
        this counter only runs when an exact figure is asked for (``Agent.budget``).
        """
        counted = self._client().messages.count_tokens(**self._base_request(messages, tools))
        return counted.input_tokens

    def _request(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> _Wire:
        return {**self._base_request(messages, tools), "max_tokens": self._max_tokens}

    def _base_request(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> _Wire:
        """The wire fields shared by ``messages.create`` and ``messages.count_tokens``."""
        request: _Wire = {"model": self._model, "messages": _to_messages(messages)}
        system = _system(messages)
        if system:
            request["system"] = system
        rendered = _to_tools(tools)
        if rendered:
            request["tools"] = rendered
        return request

    def _client(self) -> Anthropic:
        if self._client_cache is None:
            _ensure_sdk()
            from anthropic import Anthropic

            self._client_cache = Anthropic(api_key=self._api_key)
        return self._client_cache


def _ensure_sdk() -> None:
    """Confirm ``anthropic`` is importable; raise a friendly error if not."""
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise ProviderError(
            "Anthropic support needs the optional extra: install 'clawde-core[anthropic]'."
        ) from exc


def _system(messages: Sequence[Message]) -> str:
    return "\n".join(message.content for message in messages if message.role is Role.SYSTEM)


def _to_messages(messages: Sequence[Message]) -> list[_Wire]:
    out: list[_Wire] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "",
                            "content": message.content,
                        }
                    ],
                }
            )
        elif message.role is Role.ASSISTANT:
            blocks: list[_Wire] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                )
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": _user_content(message)})
    return out


def _user_content(message: Message) -> list[_Wire]:
    content: list[_Wire] = [{"type": "text", "text": message.content}]
    for image in message.images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": base64.standard_b64encode(image.data).decode("ascii"),
                },
            }
        )
    return content


def _to_tools(tools: Sequence[ToolSpec]) -> list[_Wire]:
    return [
        {"name": spec.name, "description": spec.description, "input_schema": dict(spec.parameters)}
        for spec in tools
    ]


def _to_completion(message: AnthropicMessage) -> Completion:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in message.content:
        if block.type == "text":
            texts.append(block.text)
        elif block.type == "tool_use":
            raw = block.input
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=dict(raw) if isinstance(raw, dict) else {},
                )
            )
    return Completion(text="".join(texts), tool_calls=tuple(calls), usage=_to_usage(message.usage))


def _to_usage(usage: AnthropicUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cache_read_input_tokens or 0,
    )


@PROVIDERS.register("anthropic")
def _build_anthropic() -> ModelProvider:
    """Build the Anthropic provider from settings — the ``PROVIDERS`` builder."""
    settings = get_settings()
    return AnthropicProvider(
        api_key=settings.providers.anthropic.api_key or "",
        model=settings.default_model or DEFAULT_MODEL,
    )

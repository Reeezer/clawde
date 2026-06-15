"""The OpenAI-compatible provider — OpenAI itself and any compatible endpoint.

One implementation speaks the OpenAI Chat Completions wire format, so a single
``base_url`` switch points it at ``api.openai.com`` or a local Ollama / LM Studio
/ vLLM server. Lazy-imports ``openai`` inside the call (the optional ``openai``
extra, ADR-0003). Native tool-calling is assembled from streamed deltas; weak
local models without it pair with the JSON-in-text fallback
(:mod:`clawde_core.providers.json_tools`).
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from clawde_core.config import get_settings
from clawde_core.models import (
    Completion,
    Message,
    ReasoningEffort,
    Role,
    StreamChunk,
    ToolCall,
    ToolSpec,
    Usage,
)
from clawde_core.providers import PROVIDERS
from clawde_core.providers.base import ModelProvider, ProviderError

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionUsage
    from tiktoken import Encoding

DEFAULT_MODEL = "gpt-4o-mini"
# tiktoken framing/fallback for count_tokens (ADR-0004). The gpt-4o family's 128k
# context window equals the ABC default, so context_window is inherited, not set.
_TOKENS_PER_MESSAGE = 4  # ~per-message framing overhead (role + delimiters)
_FALLBACK_ENCODING = "o200k_base"  # gpt-4o family encoding, for unknown model ids

# clawde's normalised effort → OpenAI's ``reasoning_effort``. OpenAI's reasoning
# models accept low / medium / high only — no xhigh or max — so those higher
# levels are reported unsupported rather than clamped down.
_REASONING_EFFORTS: dict[ReasoningEffort, str] = {
    ReasoningEffort.LOW: "low",
    ReasoningEffort.MEDIUM: "medium",
    ReasoningEffort.HIGH: "high",
}

# reasoning_effort is a reasoning-model feature; the gpt-3 / gpt-4 (incl. 4o)
# families don't accept it. They're denied here while every other model —
# o-series, gpt-5, and local reasoning models on a custom base_url — is allowed,
# so BYOM endpoints keep working and the backend stays the final authority.
_NON_REASONING_PREFIXES = ("gpt-3", "gpt-4", "chatgpt")


def _effort_map(model: str) -> dict[ReasoningEffort, str]:
    """The effort→native mapping for ``model`` (empty for non-reasoning families)."""
    return {} if model.startswith(_NON_REASONING_PREFIXES) else dict(_REASONING_EFFORTS)


# untyped: the openai SDK's create() takes TypedDict params; clawde builds the
# equivalent wire dicts and lets the SDK validate them at the boundary.
type _Wire = dict[str, Any]


class OpenAICompatibleProvider(ModelProvider):
    """Talks to an OpenAI-compatible Chat Completions endpoint via the ``openai`` SDK."""

    def __init__(
        self, *, api_key: str, model: str = DEFAULT_MODEL, base_url: str | None = None
    ) -> None:
        if not api_key:
            raise ProviderError(
                "OpenAI API key is missing. Set CLAWDE_PROVIDERS__OPENAI__API_KEY in your .env "
                "(any non-empty value works for a local endpoint such as Ollama)."
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client_cache: OpenAI | None = None

    @property
    def model(self) -> str:
        return self._model

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        response = self._client().chat.completions.create(**self._request(messages, tools))
        return _to_completion(response)

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        chunks = self._client().chat.completions.create(
            **self._request(messages, tools),
            stream=True,
            stream_options={"include_usage": True},
        )
        text_parts: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        usage: CompletionUsage | None = None
        for chunk in chunks:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
                yield StreamChunk(text=delta.content)
            for tool_call in delta.tool_calls or []:
                _accumulate(calls, tool_call)
        yield StreamChunk(
            completion=Completion(
                text="".join(text_parts),
                tool_calls=_assembled_calls(calls),
                usage=_to_usage(usage),
            )
        )

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        """Exact, local count via ``tiktoken`` — no API call (ADR-0004).

        Sums the encoded message contents (plus a small per-message framing
        constant) and the encoded tool schemas. Framing overhead is approximate;
        the budget is a guardrail, not a billing figure.
        """
        encoding = _encoding_for(self._model)
        total = sum(
            len(encoding.encode(message.content)) + _TOKENS_PER_MESSAGE for message in messages
        )
        total += sum(
            len(encoding.encode(spec.name + spec.description + json.dumps(spec.parameters)))
            for spec in tools
        )
        return total

    def _request(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> _Wire:
        request: _Wire = {"model": self._model, "messages": _to_messages(messages)}
        rendered = _to_tools(tools)
        if rendered:
            request["tools"] = rendered
        effort = self._resolve_effort(_effort_map(self._model), model=self._model)
        if effort is not None:
            request["reasoning_effort"] = effort
        return request

    def _client(self) -> OpenAI:
        if self._client_cache is None:
            _ensure_sdk()
            from openai import OpenAI

            self._client_cache = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client_cache


def _ensure_sdk() -> None:
    """Confirm ``openai`` is importable; raise a friendly error if not."""
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ProviderError(
            "OpenAI-compatible support needs the optional extra: install 'clawde-core[openai]'."
        ) from exc


def _ensure_tiktoken() -> None:
    """Confirm ``tiktoken`` is importable; raise a friendly error if not."""
    try:
        import tiktoken  # noqa: F401
    except ImportError as exc:
        raise ProviderError(
            "OpenAI token counting needs 'tiktoken': install 'clawde-core[openai]'."
        ) from exc


def _encoding_for(model: str) -> Encoding:
    """The model's ``tiktoken`` encoding, falling back to o200k_base for unknown ids."""
    _ensure_tiktoken()
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


def _to_messages(messages: Sequence[Message]) -> list[_Wire]:
    out: list[_Wire] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            out.append({"role": "system", "content": message.content})
        elif message.role is Role.TOOL:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id or "",
                    "content": message.content,
                }
            )
        elif message.role is Role.ASSISTANT:
            assistant: _Wire = {"role": "assistant", "content": message.content or None}
            if message.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(dict(call.arguments)),
                        },
                    }
                    for call in message.tool_calls
                ]
            out.append(assistant)
        else:
            out.append({"role": "user", "content": _user_content(message)})
    return out


def _user_content(message: Message) -> str | list[_Wire]:
    if not message.images:
        return message.content
    parts: list[_Wire] = [{"type": "text", "text": message.content}]
    for image in message.images:
        encoded = base64.standard_b64encode(image.data).decode("ascii")
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{image.mime_type};base64,{encoded}"}}
        )
    return parts


def _to_tools(tools: Sequence[ToolSpec]) -> list[_Wire]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": dict(spec.parameters),
            },
        }
        for spec in tools
    ]


def _to_completion(response: ChatCompletion) -> Completion:
    message = response.choices[0].message
    calls = tuple(
        ToolCall(
            id=call.id,
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments),
        )
        for call in message.tool_calls or []
        if call.type == "function"  # the API may also return custom tool calls; clawde ignores them
    )
    return Completion(text=message.content or "", tool_calls=calls, usage=_to_usage(response.usage))


def _accumulate(calls: dict[int, dict[str, str]], delta: Any) -> None:
    # untyped: ``delta`` is an SDK streamed tool-call fragment (ChoiceDeltaToolCall).
    entry = calls.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
    if delta.id:
        entry["id"] = delta.id
    function = delta.function
    if function is not None:
        if function.name:
            entry["name"] = function.name
        if function.arguments:
            entry["arguments"] += function.arguments


def _assembled_calls(calls: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(id=entry["id"], name=entry["name"], arguments=_parse_arguments(entry["arguments"]))
        for _index, entry in sorted(calls.items())
    )


def _parse_arguments(raw: str) -> dict[str, object]:
    """Parse a tool call's JSON-string arguments, tolerating empty or malformed input."""
    if not raw:
        return {}
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_usage(usage: CompletionUsage | None) -> Usage:
    if usage is None:
        return Usage()
    cached = 0
    details = usage.prompt_tokens_details
    if details is not None and details.cached_tokens is not None:
        cached = details.cached_tokens
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cached_tokens=cached,
    )


@PROVIDERS.register("openai")
def _build_openai() -> ModelProvider:
    """Build the OpenAI-compatible provider from settings — the ``PROVIDERS`` builder."""
    settings = get_settings()
    return OpenAICompatibleProvider(
        api_key=settings.providers.openai.api_key or "",
        model=settings.default_model or DEFAULT_MODEL,
        base_url=settings.providers.openai.base_url,
    )

"""The Gemini provider — clawde talking to Google's Gemini models.

Lazy-imports ``google-genai`` inside the call (the optional ``gemini`` extra,
ADR-0003) so the base install stays SDK-free. Translates clawde's typed messages
and tools into Gemini ``Content`` / ``Tool`` objects, calls ``generate_content``
(or ``generate_content_stream`` for SSE) with *automatic* function calling
disabled — clawde runs the loop itself — and normalises the reply (text,
function calls, token usage) back into clawde's models.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

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
    from google import genai
    from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"

# clawde's normalised effort → Gemini's ``ThinkingLevel`` name (kept as a string
# so the SDK enum stays a lazy import). thinking_level is a Gemini-3 control that
# tops out at HIGH, so XHIGH / MAX are reported unsupported rather than clamped.
# The 2.5 family steers thinking by token budget instead and so has no level map.
_THINKING_LEVELS: dict[ReasoningEffort, str] = {
    ReasoningEffort.LOW: "LOW",
    ReasoningEffort.MEDIUM: "MEDIUM",
    ReasoningEffort.HIGH: "HIGH",
}

# Model families that accept thinking_level (Gemini 3 and later). Earlier
# families (2.5, 1.5) steer thinking by token budget, which clawde doesn't drive,
# so effort on them is reported unsupported.
_THINKING_LEVEL_MODELS = ("gemini-3",)


def _effort_map(model: str) -> dict[ReasoningEffort, str]:
    """The effort→ThinkingLevel mapping for ``model`` (empty when it has no levels)."""
    return dict(_THINKING_LEVELS) if model.startswith(_THINKING_LEVEL_MODELS) else {}


class GeminiProvider(ModelProvider):
    """Talks to the Gemini Developer API via the ``google-genai`` SDK."""

    def __init__(self, *, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ProviderError(
                "Gemini API key is missing. Set CLAWDE_PROVIDERS__GEMINI__API_KEY in your .env."
            )
        self._api_key = api_key
        self._model = model
        self._client_cache: genai.Client | None = None

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        response = self._client().models.generate_content(
            model=self._model,
            contents=_to_contents(messages),
            config=_build_config(messages, tools, self._thinking_level()),
        )
        return _to_completion(response)

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        text_parts: list[str] = []
        calls: list[tuple[types.FunctionCall, bytes | None]] = []
        usage: types.GenerateContentResponseUsageMetadata | None = None
        responses = self._client().models.generate_content_stream(
            model=self._model,
            contents=_to_contents(messages),
            config=_build_config(messages, tools, self._thinking_level()),
        )
        for response in responses:
            delta = _text_of(response)
            if delta:
                text_parts.append(delta)
                yield StreamChunk(text=delta)
            calls.extend(_function_calls(response))
            if response.usage_metadata is not None:
                usage = response.usage_metadata
        yield StreamChunk(
            completion=Completion(
                text="".join(text_parts),
                tool_calls=_to_tool_calls(calls),
                usage=_to_usage(usage),
            )
        )

    def _thinking_level(self) -> str | None:
        """Resolve the current effort to a Gemini ThinkingLevel name (``None`` for OFF)."""
        return self._resolve_effort(_effort_map(self._model), model=self._model)

    def _client(self) -> genai.Client:
        if self._client_cache is None:
            _ensure_sdk()
            from google import genai

            self._client_cache = genai.Client(api_key=self._api_key)
        return self._client_cache


def _ensure_sdk() -> None:
    """Confirm ``google-genai`` is importable; raise a friendly error if not."""
    try:
        import google.genai  # noqa: F401
    except ImportError as exc:
        raise ProviderError(
            "Gemini support needs the optional extra: install 'clawde-core[gemini]'."
        ) from exc


def _build_config(
    messages: Sequence[Message], tools: Sequence[ToolSpec], level: str | None
) -> types.GenerateContentConfig:
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=_system_instruction(messages),
        tools=_to_gemini_tools(tools),  # type: ignore[arg-type]
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=_to_thinking_config(level),
    )


def _to_thinking_config(level: str | None) -> types.ThinkingConfig | None:
    """Build a Gemini ``ThinkingConfig`` from a resolved level name (``None`` for OFF)."""
    if level is None:
        return None
    from google.genai import types

    return types.ThinkingConfig(thinking_level=types.ThinkingLevel[level])


def _system_instruction(messages: Sequence[Message]) -> str | None:
    system = "\n".join(message.content for message in messages if message.role is Role.SYSTEM)
    return system or None


def _to_contents(messages: Sequence[Message]) -> list[types.Content]:
    from google.genai import types

    contents: list[types.Content] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=message.name or "", response={"output": message.content}
                        )
                    ],
                )
            )
        elif message.role is Role.ASSISTANT:
            parts: list[types.Part] = []
            if message.content:
                parts.append(types.Part.from_text(text=message.content))
            for call in message.tool_calls:
                # Replay the thought_signature Gemini 3 issued with the call, or the
                # next request is rejected (a 400 INVALID_ARGUMENT). None on 2.5 and
                # other providers' calls, which the SDK simply omits.
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call.name, args=dict(call.arguments)),
                        thought_signature=call.signature,
                    )
                )
            contents.append(types.Content(role="model", parts=parts))
        else:
            user_parts: list[types.Part] = [types.Part.from_text(text=message.content)]
            for image in message.images:
                user_parts.append(types.Part.from_bytes(data=image.data, mime_type=image.mime_type))
            contents.append(types.Content(role="user", parts=user_parts))
    return contents


def _to_gemini_tools(specs: Sequence[ToolSpec]) -> list[types.Tool] | None:
    from google.genai import types

    if not specs:
        return None
    declarations = [
        types.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            parameters=types.Schema.model_validate(dict(spec.parameters)),
        )
        for spec in specs
    ]
    return [types.Tool(function_declarations=declarations)]


def _parts_of(response: types.GenerateContentResponse) -> list[types.Part]:
    """The model turn's parts, or ``[]`` when the response carries no content."""
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None or candidate.content.parts is None:
        return []
    return list(candidate.content.parts)


def _text_of(response: types.GenerateContentResponse) -> str:
    """Concatenate the response's text parts, skipping any thought parts.

    Reads ``candidates[0].content.parts`` directly rather than the SDK's
    ``response.text`` accessor, which logs a warning whenever the response also
    carries a non-text part — i.e. on every turn the model answers with a
    ``function_call``. clawde gathers those calls separately, so that warning is
    pure noise; we sidestep it by reading the parts ourselves.
    """
    return "".join(
        part.text for part in _parts_of(response) if not part.thought and isinstance(part.text, str)
    )


def _function_calls(
    response: types.GenerateContentResponse,
) -> list[tuple[types.FunctionCall, bytes | None]]:
    """The response's function calls, each paired with its ``thought_signature``.

    Gemini 3 signs each function-call part; the signature lives on the *part*, not
    the ``FunctionCall``, and the SDK's ``response.function_calls`` accessor drops
    it. clawde reads the parts itself so the signature can ride home on the
    :class:`ToolCall` and be replayed — Gemini 3 rejects a follow-up that omits it.
    """
    return [
        (part.function_call, part.thought_signature)
        for part in _parts_of(response)
        if part.function_call is not None
    ]


def _to_completion(response: types.GenerateContentResponse) -> Completion:
    return Completion(
        text=_text_of(response),
        tool_calls=_to_tool_calls(_function_calls(response)),
        usage=_to_usage(response.usage_metadata),
    )


def _to_tool_calls(
    calls: Sequence[tuple[types.FunctionCall, bytes | None]],
) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(
            id=f"{call.name}-{index}",
            name=call.name or "",
            arguments=dict(call.args or {}),
            signature=signature,
        )
        for index, (call, signature) in enumerate(calls)
    )


def _to_usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )


@PROVIDERS.register("gemini")
def _build_gemini() -> ModelProvider:
    """Build the Gemini provider from settings — the ``PROVIDERS`` builder (ADR-0003).

    Zero-arg so the generic ``Registry`` can drive it; reads the cached
    ``get_settings()`` for the key and model. A missing key surfaces as a
    :class:`ProviderError` from :class:`GeminiProvider` at build time.
    """
    settings = get_settings()
    return GeminiProvider(
        api_key=settings.providers.gemini.api_key or "",
        model=settings.default_model or DEFAULT_MODEL,
    )

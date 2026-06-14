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

from clawde_core.models import Completion, Message, Role, StreamChunk, ToolCall, ToolSpec, Usage
from clawde_core.providers.base import ModelProvider, ProviderError

if TYPE_CHECKING:
    from google import genai
    from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"


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
            config=_build_config(messages, tools),
        )
        return _to_completion(response)

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        text_parts: list[str] = []
        calls: list[types.FunctionCall] = []
        usage: types.GenerateContentResponseUsageMetadata | None = None
        responses = self._client().models.generate_content_stream(
            model=self._model,
            contents=_to_contents(messages),
            config=_build_config(messages, tools),
        )
        for response in responses:
            delta = _text_of(response)
            if delta:
                text_parts.append(delta)
                yield StreamChunk(text=delta)
            calls.extend(response.function_calls or [])
            if response.usage_metadata is not None:
                usage = response.usage_metadata
        yield StreamChunk(
            completion=Completion(
                text="".join(text_parts),
                tool_calls=_to_tool_calls(calls),
                usage=_to_usage(usage),
            )
        )

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
    messages: Sequence[Message], tools: Sequence[ToolSpec]
) -> types.GenerateContentConfig:
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=_system_instruction(messages),
        tools=_to_gemini_tools(tools),  # type: ignore[arg-type]
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


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
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(name=call.name, args=dict(call.arguments))
                    )
                )
            contents.append(types.Content(role="model", parts=parts))
        else:
            contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=message.content)])
            )
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


def _text_of(response: types.GenerateContentResponse) -> str:
    """Concatenate the response's text parts, skipping any thought parts.

    Reads ``candidates[0].content.parts`` directly rather than the SDK's
    ``response.text`` accessor, which logs a warning whenever the response also
    carries a non-text part — i.e. on every turn the model answers with a
    ``function_call``. clawde gathers those calls separately via
    ``response.function_calls``, so that warning is pure noise; we sidestep it by
    reading the parts ourselves.
    """
    candidate = response.candidates[0] if response.candidates else None
    if candidate is None or candidate.content is None or candidate.content.parts is None:
        return ""
    texts: list[str] = []
    for part in candidate.content.parts:
        if part.thought:
            continue
        if isinstance(part.text, str):
            texts.append(part.text)
    return "".join(texts)


def _to_completion(response: types.GenerateContentResponse) -> Completion:
    return Completion(
        text=_text_of(response),
        tool_calls=_to_tool_calls(response.function_calls or []),
        usage=_to_usage(response.usage_metadata),
    )


def _to_tool_calls(function_calls: Sequence[types.FunctionCall]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(id=f"{call.name}-{index}", name=call.name or "", arguments=dict(call.args or {}))
        for index, call in enumerate(function_calls)
    )


def _to_usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )

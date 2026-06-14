"""The Gemini provider — clawde talking to Google's Gemini models.

Lazy-imports ``google-genai`` inside the call (the optional ``gemini`` extra,
ADR-0003) so the base install stays SDK-free. Translates clawde's typed messages
and tools into Gemini ``Content`` / ``Tool`` objects, calls ``generate_content``
with *automatic* function calling disabled — clawde runs the loop itself — and
normalises the reply (text, function calls, token usage) back into a
:class:`~clawde_core.models.Completion`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from clawde_core.models import Completion, Message, Role, ToolCall, ToolSpec, Usage
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
        client = self._client()
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=_system_instruction(messages),
            tools=_to_gemini_tools(tools),  # type: ignore[arg-type]
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = client.models.generate_content(
            model=self._model, contents=_to_contents(messages), config=config
        )
        return _to_completion(response)

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


def _to_completion(response: types.GenerateContentResponse) -> Completion:
    calls = response.function_calls or []
    tool_calls = tuple(
        ToolCall(id=f"{call.name}-{index}", name=call.name or "", arguments=dict(call.args or {}))
        for index, call in enumerate(calls)
    )
    return Completion(
        text=response.text or "",
        tool_calls=tool_calls,
        usage=_to_usage(response.usage_metadata),
    )


def _to_usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage:
    if metadata is None:
        return Usage()
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )

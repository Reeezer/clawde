from __future__ import annotations

import builtins
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytest

from clawde_core.models import Message, ToolCall, ToolSpec, Usage
from clawde_core.providers import gemini as gemini_module
from clawde_core.providers.base import ProviderError
from clawde_core.providers.gemini import DEFAULT_MODEL, GeminiProvider

if TYPE_CHECKING:
    from google.genai import types


class _FakeModels:
    def __init__(self, response: object = None, chunks: list[object] | None = None) -> None:
        self.response = response
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def generate_content(self, *, model: str, contents: object, config: object) -> object:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response

    def generate_content_stream(self, *, model: str, contents: object, config: object) -> object:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return iter(self.chunks or [])


class _FakeClient:
    def __init__(self, response: object = None, chunks: list[object] | None = None) -> None:
        self.models = _FakeModels(response, chunks)


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    response: object = None,
    chunks: list[object] | None = None,
) -> tuple[_FakeClient, dict[str, int]]:
    from google import genai

    client = _FakeClient(response, chunks)
    created = {"count": 0}

    def factory(**kwargs: object) -> _FakeClient:
        created["count"] += 1
        return client

    monkeypatch.setattr(genai, "Client", factory)
    return client, created


def _response(
    *,
    text: str | None = None,
    function_calls: Sequence[types.FunctionCall] = (),
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    thought: str | None = None,
) -> types.GenerateContentResponse:
    """Build a realistic Gemini response from text / thought / function-call parts.

    Faithful to the SDK shape the provider reads: text and tool calls travel as
    ``Part``s under ``candidates[0].content``, never as bare attributes — which
    is what lets these tests exercise the real ``response.function_calls`` and
    ``_text_of`` paths instead of a hand-faked ``.text``.
    """
    from google.genai import types

    parts: list[types.Part] = []
    if thought is not None:
        parts.append(types.Part(text=thought, thought=True))
    if text is not None:
        parts.append(types.Part(text=text))
    parts.extend(types.Part(function_call=call) for call in function_calls)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=usage,
    )


# --- construction & SDK guard -------------------------------------------------


def test_missing_api_key_raises() -> None:
    with pytest.raises(ProviderError, match="API key"):
        GeminiProvider(api_key="", model=DEFAULT_MODEL)


def test_ensure_sdk_passes_when_installed() -> None:
    gemini_module._ensure_sdk()  # must not raise when the SDK is installed


def test_ensure_sdk_raises_friendly_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("google"):
            raise ImportError("no google-genai")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match=r"clawde-core\[gemini\]"):
        gemini_module._ensure_sdk()


# --- message / tool conversion ------------------------------------------------


def test_to_contents_maps_every_role() -> None:
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    messages = [
        Message.system("a system prompt, sent separately"),
        Message.user("hello"),
        Message.assistant(content="let me look", tool_calls=(call,)),
        Message.tool(tool_call_id="c1", content="files listed", name="bash"),
    ]

    contents = gemini_module._to_contents(messages)

    # system is excluded (it travels as system_instruction, not as a turn)
    assert [content.role for content in contents] == ["user", "model", "tool"]


def test_system_instruction_joins_system_messages() -> None:
    assert gemini_module._system_instruction([Message.user("x")]) is None
    joined = gemini_module._system_instruction(
        [Message.system("be terse"), Message.system("be kind"), Message.user("x")]
    )
    assert joined == "be terse\nbe kind"


def test_to_gemini_tools_is_none_when_no_tools() -> None:
    assert gemini_module._to_gemini_tools([]) is None


def test_to_gemini_tools_builds_declarations() -> None:
    spec = ToolSpec(
        name="bash",
        description="run a command",
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    )

    tools = gemini_module._to_gemini_tools([spec])

    assert tools is not None
    declarations = tools[0].function_declarations
    assert declarations is not None
    assert declarations[0].name == "bash"


# --- _text_of(): read text from parts, never the warning-prone .text accessor -


def test_text_of_keeps_text_and_skips_thought_and_function_call() -> None:
    from google.genai import types

    response = _response(
        text="the answer",
        function_calls=[types.FunctionCall(name="bash", args={})],
        thought="(internal reasoning)",
    )

    assert gemini_module._text_of(response) == "the answer"


def test_text_of_returns_empty_when_no_text_parts() -> None:
    from google.genai import types

    textless = [
        types.GenerateContentResponse(candidates=None),
        types.GenerateContentResponse(candidates=[]),
        types.GenerateContentResponse(candidates=[types.Candidate(content=None)]),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=None))]
        ),
        _response(function_calls=[types.FunctionCall(name="bash", args={})]),
    ]

    assert [gemini_module._text_of(response) for response in textless] == ["", "", "", "", ""]


# --- complete(): the full call + normalisation --------------------------------


def test_complete_normalises_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import types

    response = _response(
        text="hi there",
        usage=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5, cached_content_token_count=2
        ),
    )
    client, created = _install_fake_client(monkeypatch, response)
    provider = GeminiProvider(api_key="key", model="gemini-2.5-flash")

    completion = provider.complete([Message.user("hi")], [])
    provider.complete([Message.user("hi again")], [])  # second call reuses the client

    assert completion.text == "hi there"
    assert completion.tool_calls == ()
    assert completion.usage == Usage(input_tokens=10, output_tokens=5, cached_tokens=2)
    assert created["count"] == 1  # client built once, then cached

    sent = client.models.calls[0]
    assert sent["model"] == "gemini-2.5-flash"
    config = sent["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.tools is None  # no tools were offered


def test_complete_normalises_function_calls_and_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from google.genai import types

    function_call = types.FunctionCall(name="bash", args={"command": "ls"})
    response = _response(function_calls=[function_call])
    client, _ = _install_fake_client(monkeypatch, response)
    provider = GeminiProvider(api_key="key", model="gemini-2.5-flash")
    spec = ToolSpec(
        name="bash",
        description="run a command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )

    completion = provider.complete(
        [Message.system("be helpful"), Message.user("list files")], [spec]
    )

    assert completion.text == ""
    assert [c.name for c in completion.tool_calls] == ["bash"]
    assert completion.tool_calls[0].arguments == {"command": "ls"}
    assert completion.usage == Usage()  # metadata absent -> zeros

    config = client.models.calls[0]["config"]
    assert isinstance(config, types.GenerateContentConfig)
    assert config.system_instruction == "be helpful"
    assert config.tools is not None
    afc = config.automatic_function_calling
    assert afc is not None
    assert afc.disable is True
    contents = client.models.calls[0]["contents"]
    assert isinstance(contents, list)
    assert len(contents) == 1  # only the user turn (system went to system_instruction)


def test_complete_does_not_emit_the_non_text_parts_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from google.genai import types

    # Re-arm the SDK's once-per-process guard so any stray `response.text` access
    # in the provider would log *here* — making this a real regression test, not
    # a no-op riding a flag some earlier test already tripped.
    monkeypatch.setattr(types, "_response_text_non_text_warning_logged", False)
    response = _response(function_calls=[types.FunctionCall(name="bash", args={"command": "ls"})])
    _install_fake_client(monkeypatch, response)
    provider = GeminiProvider(api_key="key", model=DEFAULT_MODEL)

    with caplog.at_level(logging.WARNING, logger="google_genai.types"):
        completion = provider.complete([Message.user("list files")], [])

    assert [call.name for call in completion.tool_calls] == ["bash"]
    assert completion.text == ""
    assert "non-text parts" not in caplog.text


# --- stream(): SSE deltas + final completion ----------------------------------


def test_stream_yields_deltas_then_final_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import types

    chunks: list[object] = [
        _response(text="Hello "),
        _response(
            text="world",
            usage=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=4, candidates_token_count=2, cached_content_token_count=0
            ),
        ),
    ]
    client, _ = _install_fake_client(monkeypatch, chunks=chunks)
    provider = GeminiProvider(api_key="key", model="gemini-2.5-flash")

    out = list(provider.stream([Message.user("hi")], []))

    assert [chunk.text for chunk in out if chunk.completion is None] == ["Hello ", "world"]
    final = out[-1].completion
    assert final is not None
    assert final.text == "Hello world"
    assert final.tool_calls == ()
    assert final.usage == Usage(input_tokens=4, output_tokens=2)
    assert client.models.calls[0]["model"] == "gemini-2.5-flash"


def test_stream_collects_function_calls_without_text(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import types

    function_call = types.FunctionCall(name="bash", args={"command": "ls"})
    chunks: list[object] = [_response(function_calls=[function_call])]
    _install_fake_client(monkeypatch, chunks=chunks)
    provider = GeminiProvider(api_key="key", model="gemini-2.5-flash")
    spec = ToolSpec(name="bash", description="run", parameters={"type": "object"})

    out = list(provider.stream([Message.user("x")], [spec]))

    assert len(out) == 1  # no text deltas, just the terminal completion
    final = out[0].completion
    assert final is not None
    assert [c.name for c in final.tool_calls] == ["bash"]
    assert final.tool_calls[0].arguments == {"command": "ls"}
    assert final.usage == Usage()


@pytest.mark.integration
def test_gemini_live_round_trip() -> None:
    from clawde_core.config import get_settings

    api_key = get_settings().providers.gemini.api_key
    if not api_key:
        pytest.skip("no Gemini API key configured")
    provider = GeminiProvider(api_key=api_key, model=DEFAULT_MODEL)

    completion = provider.complete([Message.user("Reply with exactly the word: pong")], [])

    assert "pong" in completion.text.lower()
    assert completion.usage.total > 0

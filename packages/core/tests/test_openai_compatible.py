from __future__ import annotations

import base64
import builtins
import json
from types import SimpleNamespace
from typing import Any

import pytest

from clawde_core.config import OpenAICompatibleSettings, ProvidersSettings, Settings
from clawde_core.models import ImageContent, Message, ToolCall, ToolSpec, Usage
from clawde_core.providers import PROVIDERS
from clawde_core.providers import openai_compatible as openai_module
from clawde_core.providers.base import ProviderError
from clawde_core.providers.openai_compatible import DEFAULT_MODEL, OpenAICompatibleProvider


class _FakeCompletions:
    def __init__(self, response: object = None, chunks: list[object] | None = None) -> None:
        self._response = response
        self._chunks = chunks
        # untyped: records the SDK kwargs (TypedDict wire dicts) for assertions.
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return iter(self._chunks or [])
        return self._response


class _FakeClient:
    def __init__(self, response: object = None, chunks: list[object] | None = None) -> None:
        self.completions = _FakeCompletions(response, chunks)
        self.chat = SimpleNamespace(completions=self.completions)


def _install(
    monkeypatch: pytest.MonkeyPatch, response: object = None, chunks: list[object] | None = None
) -> tuple[_FakeClient, dict[str, int]]:
    import openai

    client = _FakeClient(response, chunks)
    created = {"count": 0}

    def factory(**kwargs: object) -> _FakeClient:
        created["count"] += 1
        return client

    monkeypatch.setattr(openai, "OpenAI", factory)
    return client, created


def _usage(prompt: int = 0, completion: int = 0, cached: int | None = None) -> SimpleNamespace:
    details = SimpleNamespace(cached_tokens=cached) if cached is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details
    )


def _response(
    *,
    content: str | None = None,
    tool_calls: tuple[tuple[str, str, str], ...] = (),
    usage: object = None,
) -> SimpleNamespace:
    """A duck-typed ``ChatCompletion``: one choice, optional tool calls + usage."""
    calls = [
        SimpleNamespace(
            id=cid, type="function", function=SimpleNamespace(name=name, arguments=args)
        )
        for cid, name, args in tool_calls
    ]
    message = SimpleNamespace(content=content, tool_calls=calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _delta_chunk(
    content: str | None = None,
    tool_calls: tuple[tuple[int, str | None, str | None, str | None], ...] = (),
) -> SimpleNamespace:
    deltas = [
        SimpleNamespace(index=i, id=cid, function=SimpleNamespace(name=name, arguments=args))
        for i, cid, name, args in tool_calls
    ]
    delta = SimpleNamespace(content=content, tool_calls=deltas or None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _usage_chunk(usage: object) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=usage)


# --- construction & SDK guard -------------------------------------------------


def test_missing_api_key_raises() -> None:
    with pytest.raises(ProviderError, match="API key"):
        OpenAICompatibleProvider(api_key="")


def test_ensure_sdk_passes_when_installed() -> None:
    openai_module._ensure_sdk()


def test_ensure_sdk_raises_friendly_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openai":
            raise ImportError("no openai")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match=r"clawde-core\[openai\]"):
        openai_module._ensure_sdk()


# --- message / tool conversion ------------------------------------------------


def test_to_messages_maps_every_role() -> None:
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    messages = [
        Message.system("be helpful"),
        Message.user("hello"),
        Message.assistant(content="on it", tool_calls=(call,)),
        Message.tool(tool_call_id="c1", content="done", name="bash"),
    ]

    out = openai_module._to_messages(messages)

    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[0] == {"role": "system", "content": "be helpful"}
    assert out[1] == {"role": "user", "content": "hello"}  # plain string when no images
    assistant = out[2]
    assert assistant["content"] == "on it"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": json.dumps({"command": "ls"}),
    }
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "done"}


def test_assistant_without_tool_calls_omits_the_key() -> None:
    out = openai_module._to_messages([Message.assistant(content="just text")])
    assert out[0] == {"role": "assistant", "content": "just text"}


def test_user_content_with_images_becomes_parts() -> None:
    image = ImageContent(mime_type="image/png", data=b"PNG")

    content = openai_module._user_content(Message.user("what is this?", images=(image,)))

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    encoded = base64.standard_b64encode(b"PNG").decode("ascii")
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{encoded}"


def test_to_tools_builds_function_definitions() -> None:
    spec = ToolSpec(name="bash", description="run", parameters={"type": "object"})

    assert openai_module._to_tools([spec]) == [
        {
            "type": "function",
            "function": {"name": "bash", "description": "run", "parameters": {"type": "object"}},
        }
    ]


# --- _parse_arguments() -------------------------------------------------------


def test_parse_arguments_handles_valid_empty_malformed_and_non_dict() -> None:
    assert openai_module._parse_arguments('{"command": "ls"}') == {"command": "ls"}
    assert openai_module._parse_arguments("") == {}
    assert openai_module._parse_arguments("not json") == {}
    assert openai_module._parse_arguments("123") == {}


# --- complete(): full call + normalisation ------------------------------------


def test_complete_normalises_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(content="hi there", usage=_usage(prompt=10, completion=5, cached=2))
    client, created = _install(monkeypatch, response=response)
    provider = OpenAICompatibleProvider(api_key="key", model="gpt-test")

    completion = provider.complete([Message.system("s"), Message.user("hi")], [])
    provider.complete([Message.user("again")], [])  # reuses the cached client

    assert completion.text == "hi there"
    assert completion.tool_calls == ()
    assert completion.usage == Usage(input_tokens=10, output_tokens=5, cached_tokens=2)
    assert created["count"] == 1
    sent = client.completions.calls[0]
    assert sent["model"] == "gpt-test"
    assert "tools" not in sent


def test_complete_normalises_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _response(tool_calls=(("tc1", "bash", '{"command": "ls"}'),), usage=None)
    client, _ = _install(monkeypatch, response=response)
    provider = OpenAICompatibleProvider(api_key="key")
    spec = ToolSpec(name="bash", description="run", parameters={"type": "object"})

    completion = provider.complete([Message.user("list")], [spec])

    assert completion.text == ""
    assert [c.name for c in completion.tool_calls] == ["bash"]
    assert completion.tool_calls[0].arguments == {"command": "ls"}
    assert completion.usage == Usage()  # usage None -> zeros
    assert client.completions.calls[0]["tools"][0]["function"]["name"] == "bash"


# --- stream(): deltas + assembled tool calls + usage --------------------------


def test_stream_yields_text_deltas_then_final(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[object] = [
        _delta_chunk(content="Hello "),
        _delta_chunk(content="world"),
        _usage_chunk(_usage(prompt=4, completion=2, cached=0)),
    ]
    client, _ = _install(monkeypatch, chunks=chunks)
    provider = OpenAICompatibleProvider(api_key="key", model="gpt-test")

    out = list(provider.stream([Message.user("hi")], []))

    assert [c.text for c in out if c.completion is None] == ["Hello ", "world"]
    final = out[-1].completion
    assert final is not None
    assert final.text == "Hello world"
    assert final.tool_calls == ()
    assert final.usage == Usage(input_tokens=4, output_tokens=2)
    assert client.completions.calls[0]["stream"] is True


def test_stream_assembles_tool_calls_from_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[object] = [
        _delta_chunk(tool_calls=((0, "tc1", "bash", '{"command": '),)),
        _delta_chunk(tool_calls=((0, None, None, '"ls"}'),)),
    ]
    _install(monkeypatch, chunks=chunks)
    provider = OpenAICompatibleProvider(api_key="key")

    out = list(provider.stream([Message.user("go")], []))

    final = out[-1].completion
    assert final is not None
    assert [c.name for c in final.tool_calls] == ["bash"]
    assert final.tool_calls[0].id == "tc1"
    assert final.tool_calls[0].arguments == {"command": "ls"}


# --- context window & token counting ------------------------------------------


class _CharEncoding:
    """Stand-in tiktoken encoding: one token per character (deterministic, offline)."""

    def encode(self, text: str) -> list[int]:
        return [0] * len(text)


def test_count_tokens_sums_encoded_messages_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    import tiktoken

    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda _model: _CharEncoding())
    provider = OpenAICompatibleProvider(api_key="key", model="gpt-4o-mini")
    spec = ToolSpec(name="ab", description="cd", parameters={})  # "ab"+"cd"+"{}" -> 6 chars

    counted = provider.count_tokens([Message.user("hello")], [spec])

    assert counted == (5 + 4) + 6  # "hello" + per-message framing, then the tool schema


def test_count_tokens_falls_back_to_o200k_for_unknown_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tiktoken

    def unknown(_model: str) -> object:
        raise KeyError("unknown model")

    monkeypatch.setattr(tiktoken, "encoding_for_model", unknown)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: _CharEncoding())
    provider = OpenAICompatibleProvider(api_key="key", model="mystery-model")

    assert provider.count_tokens([Message.user("hi")], []) == 2 + 4  # 2 chars + framing


def test_ensure_tiktoken_passes_when_installed() -> None:
    openai_module._ensure_tiktoken()


def test_ensure_tiktoken_raises_friendly_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match="tiktoken"):
        openai_module._ensure_tiktoken()


# --- registry builder ---------------------------------------------------------


def test_openai_is_registered() -> None:
    assert "openai" in PROVIDERS


def test_registered_builder_builds_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        default_model="local-model",
        providers=ProvidersSettings(
            openai=OpenAICompatibleSettings(api_key="k", base_url="http://localhost:11434/v1")
        ),
    )
    monkeypatch.setattr(openai_module, "get_settings", lambda: settings)

    provider = openai_module._build_openai()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._model == "local-model"
    assert provider._base_url == "http://localhost:11434/v1"


def test_registered_builder_falls_back_to_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(providers=ProvidersSettings(openai=OpenAICompatibleSettings(api_key="k")))
    monkeypatch.setattr(openai_module, "get_settings", lambda: settings)

    provider = openai_module._build_openai()

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._model == DEFAULT_MODEL


@pytest.mark.integration
def test_openai_live_round_trip() -> None:
    from clawde_core.config import get_settings

    openai_settings = get_settings().providers.openai
    if not openai_settings.api_key:
        pytest.skip("no OpenAI API key configured")
    provider = OpenAICompatibleProvider(
        api_key=openai_settings.api_key, base_url=openai_settings.base_url
    )

    completion = provider.complete([Message.user("Reply with exactly the word: pong")], [])

    assert "pong" in completion.text.lower()
    assert completion.usage.total > 0


@pytest.mark.integration
def test_count_tokens_with_the_real_encoder() -> None:
    # No API key needed — tiktoken is local; the marker keeps the one-off vocabulary
    # download out of the default offline suite (ADR-0004).
    provider = OpenAICompatibleProvider(api_key="key", model="gpt-4o-mini")

    counted = provider.count_tokens([Message.user("hello world")], [])

    assert counted > 0

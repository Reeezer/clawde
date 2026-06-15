from __future__ import annotations

import base64
import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from clawde_core.config import AnthropicSettings, ProvidersSettings, Settings
from clawde_core.models import (
    ImageContent,
    Message,
    ReasoningEffort,
    ThinkingBlock,
    ToolCall,
    ToolSpec,
    Usage,
)
from clawde_core.providers import PROVIDERS
from clawde_core.providers import anthropic as anthropic_module
from clawde_core.providers.anthropic import DEFAULT_MODEL, AnthropicProvider
from clawde_core.providers.base import ProviderError


class _FakeMessages:
    def __init__(self, response: object = None, stream: object = None) -> None:
        self._response = response
        self._stream = stream
        # untyped: records the SDK kwargs (TypedDict wire dicts) for assertions.
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self._response

    def stream(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return self._stream


class _FakeClient:
    def __init__(self, response: object = None, stream: object = None) -> None:
        self.messages = _FakeMessages(response, stream)


def _install(
    monkeypatch: pytest.MonkeyPatch, response: object = None, stream: object = None
) -> tuple[_FakeClient, dict[str, int]]:
    import anthropic

    client = _FakeClient(response, stream)
    created = {"count": 0}

    def factory(**kwargs: object) -> _FakeClient:
        created["count"] += 1
        return client

    monkeypatch.setattr(anthropic, "Anthropic", factory)
    return client, created


def _message(
    *,
    text: str | None = None,
    tool_calls: tuple[tuple[str, str, dict[str, object]], ...] = (),
    usage: object = None,
) -> SimpleNamespace:
    """A duck-typed stand-in for ``anthropic.types.Message`` (content blocks + usage)."""
    content: list[SimpleNamespace] = []
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    for call_id, name, args in tool_calls:
        content.append(SimpleNamespace(type="tool_use", id=call_id, name=name, input=args))
    return SimpleNamespace(
        content=content,
        usage=usage
        or SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=None),
    )


class _FakeStream:
    def __init__(self, texts: list[str], final: object) -> None:
        self._texts = texts
        self._final = final

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @property
    def text_stream(self) -> object:
        return iter(self._texts)

    def get_final_message(self) -> object:
        return self._final


# --- construction & SDK guard -------------------------------------------------


def test_missing_api_key_raises() -> None:
    with pytest.raises(ProviderError, match="API key"):
        AnthropicProvider(api_key="")


def test_ensure_sdk_passes_when_installed() -> None:
    anthropic_module._ensure_sdk()


def test_ensure_sdk_raises_friendly_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            raise ImportError("no anthropic")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match=r"clawde-core\[anthropic\]"):
        anthropic_module._ensure_sdk()


# --- message / tool conversion ------------------------------------------------


def test_to_messages_maps_every_non_system_role() -> None:
    image = ImageContent(mime_type="image/png", data=b"PNG")
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    messages = [
        Message.system("system, sent separately"),
        Message.user("hello", images=(image,)),
        Message.assistant(content="let me look", tool_calls=(call,)),
        Message.tool(tool_call_id="c1", content="files listed", name="bash"),
    ]

    out = anthropic_module._to_messages(messages)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]  # system excluded
    user_blocks = out[0]["content"]
    assert user_blocks[0] == {"type": "text", "text": "hello"}  # text first
    assert user_blocks[1]["type"] == "image"  # image follows
    assert user_blocks[1]["source"]["data"] == base64.standard_b64encode(b"PNG").decode("ascii")
    assistant_blocks = out[1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "let me look"}
    assert assistant_blocks[1] == {
        "type": "tool_use",
        "id": "c1",
        "name": "bash",
        "input": {"command": "ls"},
    }
    assert out[2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": "files listed",
    }


def test_system_joins_system_messages() -> None:
    assert anthropic_module._system([Message.user("x")]) == ""
    assert (
        anthropic_module._system([Message.system("be terse"), Message.system("be kind")])
        == "be terse\nbe kind"
    )


def test_to_tools_builds_definitions() -> None:
    spec = ToolSpec(
        name="bash",
        description="run a command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )

    assert anthropic_module._to_tools([spec]) == [
        {
            "name": "bash",
            "description": "run a command",
            "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
        }
    ]


# --- complete(): the full call + normalisation --------------------------------


def test_complete_normalises_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _message(
        text="hi there",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=2),
    )
    client, created = _install(monkeypatch, response=response)
    provider = AnthropicProvider(api_key="key", model="claude-test")

    completion = provider.complete([Message.system("be helpful"), Message.user("hi")], [])
    provider.complete([Message.user("again")], [])  # reuses the cached client

    assert completion.text == "hi there"
    assert completion.tool_calls == ()
    assert completion.usage == Usage(input_tokens=10, output_tokens=5, cached_tokens=2)
    assert created["count"] == 1
    sent = client.messages.calls[0]
    assert sent["model"] == "claude-test"
    assert sent["max_tokens"] == anthropic_module.DEFAULT_MAX_TOKENS
    assert sent["system"] == "be helpful"
    assert "tools" not in sent  # none offered


def test_complete_normalises_tool_use_and_request(monkeypatch: pytest.MonkeyPatch) -> None:
    response = _message(tool_calls=(("tu1", "bash", {"command": "ls"}),))
    client, _ = _install(monkeypatch, response=response)
    provider = AnthropicProvider(api_key="key")
    spec = ToolSpec(name="bash", description="run", parameters={"type": "object"})

    completion = provider.complete([Message.user("list files")], [spec])

    assert completion.text == ""
    assert [c.name for c in completion.tool_calls] == ["bash"]
    assert completion.tool_calls[0].arguments == {"command": "ls"}
    assert completion.usage == Usage()  # cache_read None -> 0
    sent = client.messages.calls[0]
    assert "system" not in sent  # no system message present
    assert sent["tools"][0]["name"] == "bash"


# --- stream(): SSE deltas + final completion ----------------------------------


def test_stream_yields_deltas_then_final_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    final = _message(
        text="Hello world",
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, cache_read_input_tokens=0),
    )
    client, _ = _install(monkeypatch, stream=_FakeStream(["Hello ", "world"], final))
    provider = AnthropicProvider(api_key="key", model="claude-test")

    out = list(provider.stream([Message.user("hi")], []))

    assert [chunk.text for chunk in out if chunk.completion is None] == ["Hello ", "world"]
    completion = out[-1].completion
    assert completion is not None
    assert completion.text == "Hello world"
    assert completion.usage == Usage(input_tokens=4, output_tokens=2)
    assert client.messages.calls[0]["model"] == "claude-test"


# --- registry builder ---------------------------------------------------------


def test_anthropic_is_registered() -> None:
    assert "anthropic" in PROVIDERS


def test_registered_builder_builds_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        default_model="claude-custom",
        providers=ProvidersSettings(anthropic=AnthropicSettings(api_key="secret-key")),
    )
    monkeypatch.setattr(anthropic_module, "get_settings", lambda: settings)

    provider = anthropic_module._build_anthropic()

    assert isinstance(provider, AnthropicProvider)
    assert provider._model == "claude-custom"


def test_registered_builder_falls_back_to_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(providers=ProvidersSettings(anthropic=AnthropicSettings(api_key="k")))
    monkeypatch.setattr(anthropic_module, "get_settings", lambda: settings)

    provider = anthropic_module._build_anthropic()

    assert isinstance(provider, AnthropicProvider)
    assert provider._model == DEFAULT_MODEL


# --- reasoning effort + thinking blocks ---------------------------------------


def test_off_effort_sends_no_reasoning_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _install(monkeypatch, response=_message(text="ok"))
    provider = AnthropicProvider(api_key="key")  # OFF by default

    provider.complete([Message.user("hi")], [])

    sent = client.messages.calls[0]
    assert "thinking" not in sent
    assert "output_config" not in sent


def test_effort_adds_adaptive_thinking_and_native_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _install(monkeypatch, response=_message(text="ok"))
    provider = AnthropicProvider(api_key="key")
    provider.set_reasoning_effort(ReasoningEffort.MAX)  # Anthropic maps every level 1:1

    provider.complete([Message.user("hi")], [])

    sent = client.messages.calls[0]
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "max"}


def test_effort_on_a_model_without_support_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, response=_message(text="ok"))
    provider = AnthropicProvider(api_key="key", model="claude-3-5-haiku-latest")
    provider.set_reasoning_effort(ReasoningEffort.HIGH)  # older Claude has no effort control

    with pytest.raises(ProviderError, match="does not support reasoning effort"):
        provider.complete([Message.user("hi")], [])


def test_complete_captures_thinking_and_redacted_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="step by step", signature="sig-abc"),
            SimpleNamespace(type="redacted_thinking", data="opaque"),
            SimpleNamespace(type="text", text="the answer"),
        ],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0),
    )
    _install(monkeypatch, response=response)
    provider = AnthropicProvider(api_key="key")

    completion = provider.complete([Message.user("hi")], [])

    assert completion.text == "the answer"
    assert completion.thinking == (
        ThinkingBlock(text="step by step", signature="sig-abc"),
        ThinkingBlock(redacted_data="opaque"),
    )


def test_to_messages_replays_thinking_blocks_first() -> None:
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    thinking = (
        ThinkingBlock(text="reason", signature="sig-1"),
        ThinkingBlock(redacted_data="opaque"),
    )
    message = Message.assistant(content="here", tool_calls=(call,), thinking=thinking)

    blocks = anthropic_module._to_messages([message])[0]["content"]

    # thinking (and redacted) blocks must lead, before text and tool_use
    assert blocks[0] == {"type": "thinking", "thinking": "reason", "signature": "sig-1"}
    assert blocks[1] == {"type": "redacted_thinking", "data": "opaque"}
    assert blocks[2] == {"type": "text", "text": "here"}
    assert blocks[3]["type"] == "tool_use"


@pytest.mark.integration
def test_anthropic_live_round_trip() -> None:
    from clawde_core.config import get_settings

    api_key = get_settings().providers.anthropic.api_key
    if not api_key:
        pytest.skip("no Anthropic API key configured")
    provider = AnthropicProvider(api_key=api_key)

    completion = provider.complete([Message.user("Reply with exactly the word: pong")], [])

    assert "pong" in completion.text.lower()
    assert completion.usage.total > 0

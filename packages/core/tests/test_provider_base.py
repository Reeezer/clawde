from __future__ import annotations

from collections.abc import Sequence

import pytest

from clawde_core.models import Completion, ImageContent, Message, ReasoningEffort, ToolSpec
from clawde_core.providers.base import DEFAULT_CONTEXT_WINDOW, ModelProvider, ProviderError


class _TextOnlyProvider(ModelProvider):
    """Minimal provider with no image support: guards before answering."""

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        self._reject_image_input(messages)
        return Completion(text="ok")


def test_text_only_provider_rejects_image_input() -> None:
    provider = _TextOnlyProvider()
    image = ImageContent(mime_type="image/png", data=b"x")

    with pytest.raises(ProviderError, match="does not support image input"):
        provider.complete([Message.user("look", images=(image,))], [])


def test_guard_is_a_no_op_without_images() -> None:
    provider = _TextOnlyProvider()

    completion = provider.complete([Message.user("hi")], [])

    assert completion.text == "ok"


def test_context_window_defaults_to_the_conservative_fallback() -> None:
    assert _TextOnlyProvider().context_window == DEFAULT_CONTEXT_WINDOW


def test_count_tokens_estimates_about_four_chars_per_token() -> None:
    provider = _TextOnlyProvider()
    messages = [Message.user("x" * 40)]
    tools = [ToolSpec(name="echo", description="run", parameters={})]

    # text = 40 chars + len("echo" + "run" + "{}") = 49; 49 // 4 + 1 message = 13
    assert provider.count_tokens(messages, tools) == 13


def test_reasoning_effort_defaults_off_and_is_settable() -> None:
    provider = _TextOnlyProvider()

    assert provider.reasoning_effort is ReasoningEffort.OFF
    provider.set_reasoning_effort(ReasoningEffort.XHIGH)
    updated: ReasoningEffort = provider.reasoning_effort
    assert updated is ReasoningEffort.XHIGH


def test_resolve_effort_returns_none_when_off() -> None:
    provider = _TextOnlyProvider()  # OFF by default

    assert provider._resolve_effort({ReasoningEffort.HIGH: "high"}, model="m") is None


def test_resolve_effort_maps_a_supported_level() -> None:
    provider = _TextOnlyProvider()
    provider.set_reasoning_effort(ReasoningEffort.HIGH)

    assert provider._resolve_effort({ReasoningEffort.HIGH: "high"}, model="m") == "high"


def test_resolve_effort_rejects_an_unsupported_level() -> None:
    provider = _TextOnlyProvider()
    provider.set_reasoning_effort(ReasoningEffort.MAX)

    with pytest.raises(ProviderError, match="it supports: off, low, high"):
        provider._resolve_effort(
            {ReasoningEffort.LOW: "low", ReasoningEffort.HIGH: "high"}, model="gemini-3-pro"
        )


def test_resolve_effort_reports_no_support_for_an_empty_map() -> None:
    provider = _TextOnlyProvider()
    provider.set_reasoning_effort(ReasoningEffort.LOW)

    with pytest.raises(ProviderError, match="does not support reasoning effort; use effort 'off'"):
        provider._resolve_effort({}, model="gpt-4o-mini")

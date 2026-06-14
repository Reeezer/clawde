from __future__ import annotations

from collections.abc import Sequence

import pytest

from clawde_core.models import Completion, ImageContent, Message, ToolSpec
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

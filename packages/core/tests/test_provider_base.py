from __future__ import annotations

from collections.abc import Sequence

import pytest

from clawde_core.models import Completion, ImageContent, Message, ToolSpec
from clawde_core.providers.base import ModelProvider, ProviderError


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

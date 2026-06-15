from __future__ import annotations

from clawde_core.providers.anthropic import AnthropicProvider
from clawde_core.providers.gemini import GeminiProvider
from clawde_core.providers.openai_compatible import OpenAICompatibleProvider


def test_anthropic_exposes_its_model() -> None:
    assert AnthropicProvider(api_key="x", model="claude-test").model == "claude-test"


def test_gemini_exposes_its_model() -> None:
    assert GeminiProvider(api_key="x", model="gemini-test").model == "gemini-test"


def test_openai_compatible_exposes_its_model() -> None:
    assert OpenAICompatibleProvider(api_key="x", model="gpt-test").model == "gpt-test"

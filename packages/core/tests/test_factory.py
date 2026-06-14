from __future__ import annotations

import pytest

from clawde_core.config import get_settings
from clawde_core.providers import factory
from clawde_core.providers.anthropic import AnthropicProvider
from clawde_core.providers.gemini import GeminiProvider
from clawde_core.providers.json_tools import JsonToolCallingProvider
from clawde_core.providers.openai_compatible import OpenAICompatibleProvider
from clawde_core.registry import RegistryError


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    # The factory and each provider builder read the same cached get_settings();
    # clearing before every test lets per-test env vars take effect cleanly.
    get_settings.cache_clear()


def test_build_provider_resolves_the_default_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("CLAWDE_PROVIDERS__GEMINI__API_KEY", "k")

    assert isinstance(factory.build_provider(), GeminiProvider)


def test_build_provider_honours_an_explicit_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_DEFAULT_PROVIDER", "gemini")  # overridden by the argument
    monkeypatch.setenv("CLAWDE_PROVIDERS__ANTHROPIC__API_KEY", "k")

    assert isinstance(factory.build_provider("anthropic"), AnthropicProvider)


def test_build_provider_applies_a_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_PROVIDERS__OPENAI__API_KEY", "k")

    provider = factory.build_provider("openai", model="my-model")

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._model == "my-model"


def test_build_provider_unknown_name_lists_known_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_DEFAULT_PROVIDER", "nope")

    with pytest.raises(RegistryError, match="anthropic, gemini, openai"):
        factory.build_provider()


def test_build_provider_wraps_in_json_fallback_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAWDE_DEFAULT_PROVIDER", "gemini")
    monkeypatch.setenv("CLAWDE_PROVIDERS__GEMINI__API_KEY", "k")
    monkeypatch.setenv("CLAWDE_TOOL_CALLING", "json")

    provider = factory.build_provider()

    assert isinstance(provider, JsonToolCallingProvider)
    assert isinstance(provider._inner, GeminiProvider)  # native provider wrapped, not replaced

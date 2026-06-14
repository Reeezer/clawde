from __future__ import annotations

import pytest

from clawde_core.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_defaults() -> None:
    settings = Settings()
    assert settings.default_provider == "anthropic"
    assert settings.default_model is None


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("CLAWDE_DEFAULT_MODEL", "gpt-5")
    settings = Settings()
    assert settings.default_provider == "openai"
    assert settings.default_model == "gpt-5"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_provider_api_key_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from the developer's real .env so the test is deterministic.
    monkeypatch.delenv("CLAWDE_PROVIDERS__GEMINI__API_KEY", raising=False)
    settings = Settings()
    assert settings.providers.gemini.api_key is None


def test_provider_api_key_from_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_PROVIDERS__GEMINI__API_KEY", "secret-123")
    settings = Settings()
    assert settings.providers.gemini.api_key == "secret-123"

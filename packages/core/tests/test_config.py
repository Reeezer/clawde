from __future__ import annotations

import pytest

from clawde_core.config import Settings, ToolCalling, get_settings


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


def test_tool_calling_defaults_to_native() -> None:
    assert Settings().tool_calling is ToolCalling.NATIVE


def test_tool_calling_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_TOOL_CALLING", "json")
    assert Settings().tool_calling is ToolCalling.JSON


def test_anthropic_and_openai_settings_from_nested_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_PROVIDERS__ANTHROPIC__API_KEY", "ant-123")
    monkeypatch.setenv("CLAWDE_PROVIDERS__OPENAI__API_KEY", "oai-123")
    monkeypatch.setenv("CLAWDE_PROVIDERS__OPENAI__BASE_URL", "http://localhost:11434/v1")
    settings = Settings()
    assert settings.providers.anthropic.api_key == "ant-123"
    assert settings.providers.openai.api_key == "oai-123"
    assert settings.providers.openai.base_url == "http://localhost:11434/v1"

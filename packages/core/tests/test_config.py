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

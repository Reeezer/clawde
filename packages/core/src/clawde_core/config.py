"""Runtime configuration, loaded from the environment and ``.env``.

Settings are read from ``CLAWDE_``-prefixed variables, with ``__`` separating
nested levels — e.g. ``CLAWDE_DEFAULT_PROVIDER=openai`` or
``CLAWDE_PROVIDERS__GEMINI__API_KEY=...``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeminiSettings(BaseModel):
    """Configuration for the Gemini provider (``CLAWDE_PROVIDERS__GEMINI__*``)."""

    api_key: str | None = None


class ProvidersSettings(BaseModel):
    """Per-provider configuration; grows as providers land (Phase 2)."""

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)


class Settings(BaseSettings):
    """Top-level clawde settings."""

    model_config = SettingsConfigDict(
        env_prefix="CLAWDE_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    default_provider: str = "anthropic"
    default_model: str | None = None
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()

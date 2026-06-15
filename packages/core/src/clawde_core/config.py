"""Runtime configuration, loaded from the environment and ``.env``.

Settings are read from ``CLAWDE_``-prefixed variables, with ``__`` separating
nested levels — e.g. ``CLAWDE_DEFAULT_PROVIDER=openai`` or
``CLAWDE_PROVIDERS__GEMINI__API_KEY=...``.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from clawde_core.models import ReasoningEffort


class ToolCalling(StrEnum):
    """How clawde asks a model to call tools.

    ``native`` uses the provider's own function-calling; ``json`` uses the
    JSON-in-text fallback (:mod:`clawde_core.providers.json_tools`) for models —
    typically local or weaker ones — that lack native tool support.
    """

    NATIVE = "native"
    JSON = "json"


class CompactionStrategy(StrEnum):
    """How clawde shrinks a conversation that nears the context window (ADR-0005).

    ``summarise`` replaces old turns with a model-written recap; ``off`` disables
    compaction so the loop never makes a mid-turn summarisation call.
    """

    OFF = "off"
    SUMMARISE = "summarise"


class GeminiSettings(BaseModel):
    """Configuration for the Gemini provider (``CLAWDE_PROVIDERS__GEMINI__*``)."""

    api_key: str | None = None


class AnthropicSettings(BaseModel):
    """Configuration for the Anthropic provider (``CLAWDE_PROVIDERS__ANTHROPIC__*``)."""

    api_key: str | None = None


class OpenAICompatibleSettings(BaseModel):
    """Config for the OpenAI-compatible provider (``CLAWDE_PROVIDERS__OPENAI__*``).

    One implementation serves OpenAI itself and every OpenAI-compatible endpoint
    (Ollama, LM Studio, vLLM, …); ``base_url`` points it at the chosen backend
    (e.g. ``http://localhost:11434/v1`` for Ollama).
    """

    api_key: str | None = None
    base_url: str | None = None


class ProvidersSettings(BaseModel):
    """Per-provider configuration."""

    gemini: GeminiSettings = Field(default_factory=GeminiSettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    openai: OpenAICompatibleSettings = Field(default_factory=OpenAICompatibleSettings)


class CompactionSettings(BaseModel):
    """Conversation compaction (``CLAWDE_COMPACTION__*``, ADR-0005).

    ``threshold`` is the share of the context window (0–1) at which the loop
    compacts before its next call; ``keep_recent_turns`` is how many recent turns
    the ``summarise`` strategy preserves verbatim.
    """

    strategy: CompactionStrategy = CompactionStrategy.SUMMARISE
    threshold: float = 0.8
    keep_recent_turns: int = 3


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
    tool_calling: ToolCalling = ToolCalling.NATIVE
    default_reasoning_effort: ReasoningEffort = ReasoningEffort.OFF
    providers: ProvidersSettings = Field(default_factory=ProvidersSettings)
    compaction: CompactionSettings = Field(default_factory=CompactionSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()

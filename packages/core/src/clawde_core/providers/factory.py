"""The provider factory — resolves ``Settings`` to a built ``ModelProvider``.

``build_provider`` reads the cached ``get_settings()`` for the chosen provider
name (the CLI's ``--provider`` overriding ``default_provider``), looks it up in
the ``PROVIDERS`` registry, and — when ``tool_calling`` is ``json`` — wraps the
result in the JSON-in-text fallback. The loop and CLI depend only on this
function, never on a concrete provider (ADR-0003).
"""

from __future__ import annotations

from clawde_core.config import ToolCalling, get_settings
from clawde_core.providers import PROVIDERS

# Importing the impls runs their ``@PROVIDERS.register(...)`` side effect. Each is
# import-time SDK-free (the SDK is lazy-imported inside the impl), so this stays
# cheap and offline-safe.
from clawde_core.providers import anthropic as _anthropic  # noqa: F401
from clawde_core.providers import gemini as _gemini  # noqa: F401
from clawde_core.providers import openai_compatible as _openai_compatible  # noqa: F401
from clawde_core.providers.base import ModelProvider
from clawde_core.providers.json_tools import JsonToolCallingProvider


def build_provider(provider: str | None = None, model: str | None = None) -> ModelProvider:
    """Build the configured provider, optionally overriding its name and model.

    ``provider`` (e.g. the CLI's ``--provider``) overrides
    ``Settings.default_provider``; ``model`` overrides ``Settings.default_model``
    for this process. An unknown name raises ``RegistryError`` listing the known
    providers. When ``Settings.tool_calling`` is ``json``, the built provider is
    wrapped in the JSON-in-text fallback so a model without native tool support
    can still drive the loop.
    """
    settings = get_settings()
    if model is not None:
        settings.default_model = model
    built = PROVIDERS.create(provider or settings.default_provider)
    if settings.tool_calling is ToolCalling.JSON:
        return JsonToolCallingProvider(built)
    return built

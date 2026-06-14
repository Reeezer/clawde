"""clawde_core — the bring-your-own-model agent engine.

Houses the agent loop, the provider abstraction (BYOM), the tool system, and
context management. Everything swappable lives behind an ABC + a named
:class:`Registry`; see ``CLAUDE.md`` and ``docs/adr/`` for the architecture.
"""

from __future__ import annotations

from clawde_core.config import Settings, get_settings
from clawde_core.loop import Agent, AgentError, Turn
from clawde_core.models import (
    Completion,
    Message,
    Role,
    StreamChunk,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
from clawde_core.providers.base import ModelProvider, ProviderError
from clawde_core.registry import Registry, RegistryError
from clawde_core.tools.base import Tool

__version__ = "0.0.1"

__all__ = [
    "Agent",
    "AgentError",
    "Completion",
    "Message",
    "ModelProvider",
    "ProviderError",
    "Registry",
    "RegistryError",
    "Role",
    "Settings",
    "StreamChunk",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
    "__version__",
    "get_settings",
]

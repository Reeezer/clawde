"""The summarising compactor — replace old turns with a model-written recap.

``strategy = "summarise"`` (the default): keep the most recent ``keep_recent_turns``
turns intact and replace everything older with a short synthetic exchange — a user
"summarise so far" request and the model's recap — so the conversation keeps a
valid shape (starts at a user turn, never splits a tool call) while shrinking. The
recap comes from one direct :meth:`~clawde_core.providers.base.ModelProvider.complete`
call, not through the agent loop, so compaction never recurses (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Sequence

from clawde_core.config import get_settings
from clawde_core.context.compaction import COMPACTORS
from clawde_core.context.compaction.base import Compactor
from clawde_core.context.history import History
from clawde_core.models import Message
from clawde_core.providers.base import ModelProvider

DEFAULT_KEEP_RECENT_TURNS = 3

_SUMMARY_SYSTEM = (
    "You compress a coding agent's conversation so it can continue in a smaller "
    "context window. Write a dense summary that preserves the user's goal, the "
    "decisions and actions taken, the files and commands touched, key facts and "
    "values discovered, and what still remains to be done. Reply with only the "
    "summary."
)
_SUMMARY_REQUEST = "Summarise our conversation so far so we can continue with less context:"
_RECAP_PREAMBLE = "Summary of the earlier conversation (compacted to save context):"


class SummarisingCompactor(Compactor):
    """Keeps the recent turns; replaces older ones with a model-written recap."""

    def __init__(self, keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS) -> None:
        self._keep_recent_turns = keep_recent_turns

    def compact(self, history: History, *, provider: ModelProvider) -> History:
        turns = history.turns()
        if len(turns) <= self._keep_recent_turns:
            return history  # nothing old enough to summarise — leave it unchanged
        older = turns[: -self._keep_recent_turns]
        kept = turns[-self._keep_recent_turns :]
        recap = (
            Message.user(_SUMMARY_REQUEST),
            Message.assistant(f"{_RECAP_PREAMBLE}\n\n{self._summarise(older, provider)}"),
        )
        kept_messages = [message for turn in kept for message in turn]
        return History([*recap, *kept_messages])

    def _summarise(self, older: Sequence[Sequence[Message]], provider: ModelProvider) -> str:
        reply = provider.complete(
            [
                Message.system(_SUMMARY_SYSTEM),
                Message.user(f"{_SUMMARY_REQUEST}\n\n{_render(older)}"),
            ],
            [],
        )
        return reply.text.strip()


def _render(turns: Sequence[Sequence[Message]]) -> str:
    """Render the turns being summarised as a plain, role-labelled transcript."""
    return "\n".join(_render_message(message) for turn in turns for message in turn)


def _render_message(message: Message) -> str:
    lines = [f"{message.role.value}: {message.content}".rstrip()]
    for call in message.tool_calls:
        lines.append(f"  -> calls {call.name}({_render_args(call.arguments)})")
    return "\n".join(lines)


def _render_args(arguments: dict[str, object]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


@COMPACTORS.register("summarise")
def _build_summarise() -> Compactor:
    """Build the summarising compactor from settings — the ``COMPACTORS`` builder."""
    return SummarisingCompactor(keep_recent_turns=get_settings().compaction.keep_recent_turns)

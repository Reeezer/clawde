"""The conversation history store — owner of the running message list.

The loop used to keep ``self._history: list[Message]`` inline and grow it
unbounded. :class:`History` gives those messages a single, typed home so token
budgeting and (next) compaction have something concrete to measure and rewrite,
while the loop stays focused on driving turns.
"""

from __future__ import annotations

from collections.abc import Iterable

from clawde_core.models import Message


class History:
    """An ordered, append-only-from-the-loop record of the conversation.

    Holds only the dialogue messages — the system prompt is assembled around it
    by the loop, not stored here.
    """

    def __init__(self, messages: Iterable[Message] = ()) -> None:
        self._messages: list[Message] = list(messages)

    def append(self, message: Message) -> None:
        """Add one message to the end of the conversation."""
        self._messages.append(message)

    @property
    def messages(self) -> tuple[Message, ...]:
        """An immutable snapshot of the full conversation, oldest first."""
        return tuple(self._messages)

    def since(self, start: int) -> tuple[Message, ...]:
        """The messages appended at or after index ``start`` (e.g. one turn)."""
        return tuple(self._messages[start:])

    def truncate(self, length: int) -> None:
        """Drop every message from index ``length`` on — e.g. to roll back an
        interrupted turn so the conversation has no half-built tail."""
        del self._messages[length:]

    def __len__(self) -> int:
        return len(self._messages)

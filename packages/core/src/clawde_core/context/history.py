"""The conversation history store — owner of the running message list.

The loop used to keep ``self._history: list[Message]`` inline and grow it
unbounded. :class:`History` gives those messages a single, typed home so token
budgeting and compaction have something concrete to measure and rewrite, while
the loop stays focused on driving turns. :meth:`turns` exposes the turn
boundaries compaction cuts on, so a tool result is never split from its call.
"""

from __future__ import annotations

from collections.abc import Iterable

from clawde_core.models import Message, Role


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

    def turns(self) -> tuple[tuple[Message, ...], ...]:
        """The conversation grouped into turns, each starting at a user message.

        A turn is a user message and every assistant / tool message that follows
        it up to (not including) the next user message — the unit compaction keeps
        or summarises whole, so a tool result never loses the tool call it answers.
        Any messages before the first user message form the leading group.
        """
        turns: list[list[Message]] = []
        for message in self._messages:
            if message.role is Role.USER or not turns:
                turns.append([message])
            else:
                turns[-1].append(message)
        return tuple(tuple(turn) for turn in turns)

    def __len__(self) -> int:
        return len(self._messages)

from __future__ import annotations

from collections.abc import Sequence

from clawde_core.config import CompactionSettings, CompactionStrategy, Settings
from clawde_core.context.compaction.factory import build_compactor
from clawde_core.context.compaction.off import NoOpCompactor
from clawde_core.context.compaction.summarise import SummarisingCompactor
from clawde_core.context.history import History
from clawde_core.models import Completion, Message, ToolCall, ToolSpec
from clawde_core.providers.base import ModelProvider


class FakeSummariser(ModelProvider):
    """Returns a fixed recap and records the conversation it was asked to summarise."""

    def __init__(self, summary: str = "SUMMARY") -> None:
        self._summary = summary
        self.received: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        self.received.append(tuple(messages))
        return Completion(text=self._summary)


def _conversation(n_turns: int) -> list[Message]:
    """``n_turns`` simple user/assistant exchanges."""
    messages: list[Message] = []
    for i in range(n_turns):
        messages.append(Message.user(f"u{i}"))
        messages.append(Message.assistant(f"a{i}"))
    return messages


def test_noop_compactor_returns_the_same_history_and_never_calls_the_model() -> None:
    history = History(_conversation(5))
    provider = FakeSummariser()

    result = NoOpCompactor().compact(history, provider=provider)

    assert result is history  # the loop detects the no-op by identity
    assert provider.received == []


def test_summarise_leaves_a_short_history_unchanged() -> None:
    history = History(_conversation(2))
    provider = FakeSummariser()

    result = SummarisingCompactor(keep_recent_turns=3).compact(history, provider=provider)

    assert result is history  # nothing old enough to summarise -> identity, no model call
    assert provider.received == []


def test_summarise_replaces_old_turns_with_a_recap_and_keeps_recent() -> None:
    history = History(_conversation(5))  # turns u0..u4
    provider = FakeSummariser(summary="recap text")

    result = SummarisingCompactor(keep_recent_turns=2).compact(history, provider=provider)

    messages = result.messages
    assert result is not history
    # a 2-message recap pair, then the last 2 turns intact (u3,a3,u4,a4)
    assert len(messages) == 6
    assert messages[0].role.value == "user"
    assert messages[1].role.value == "assistant"
    assert "recap text" in messages[1].content
    assert [m.content for m in messages[2:]] == ["u3", "a3", "u4", "a4"]


def test_summarise_sends_only_the_older_turns_to_the_model() -> None:
    history = History(_conversation(4))
    provider = FakeSummariser()

    SummarisingCompactor(keep_recent_turns=1).compact(history, provider=provider)

    request = provider.received[0][-1].content  # the user message of the summary request
    assert "u0" in request and "u2" in request  # older turns are summarised
    assert "u3" not in request  # the kept recent turn is not


def test_summarise_renders_tool_calls_and_results_into_the_transcript() -> None:
    history = History(
        [
            Message.user("do it"),
            Message.assistant(
                tool_calls=(ToolCall(id="c1", name="bash", arguments={"command": "ls"}),)
            ),
            Message.tool(tool_call_id="c1", content="file.py", name="bash"),
            Message.user("recent"),
            Message.assistant("recent reply"),
        ]
    )
    provider = FakeSummariser()

    SummarisingCompactor(keep_recent_turns=1).compact(history, provider=provider)

    request = provider.received[0][-1].content
    assert "calls bash(" in request  # the tool call is described
    assert "file.py" in request  # the tool result is part of what gets summarised


def test_build_compactor_defaults_to_the_summarising_strategy() -> None:
    assert isinstance(build_compactor(Settings()), SummarisingCompactor)


def test_build_compactor_honours_the_off_strategy() -> None:
    settings = Settings(compaction=CompactionSettings(strategy=CompactionStrategy.OFF))

    assert isinstance(build_compactor(settings), NoOpCompactor)

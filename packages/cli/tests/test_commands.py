from __future__ import annotations

import io
from collections.abc import Iterator, Sequence

from clawde_core.context.compaction.base import Compactor
from clawde_core.context.compaction.summarise import SummarisingCompactor
from clawde_core.loop import Agent
from clawde_core.models import Completion, Message, ReasoningEffort, StreamChunk, ToolSpec, Usage
from clawde_core.providers.base import ModelProvider
from rich.console import Console

from clawde_cli.commands import CommandResult, dispatch, registered_commands
from clawde_cli.session import Session


class _FakeProvider(ModelProvider):
    def __init__(self, completions: Sequence[Completion] = (), *, recap: str = "recap") -> None:
        self._queue = list(completions)
        self._recap = recap

    @property
    def model(self) -> str:
        return "fake-model"

    @property
    def context_window(self) -> int:
        return 1000

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        return Completion(text=self._recap)  # the summariser's one direct call

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        completion = self._queue.pop(0)
        if completion.text:
            yield StreamChunk(text=completion.text)
        yield StreamChunk(completion=completion)


def _session(
    completions: Sequence[Completion] = (), *, compactor: Compactor | None = None
) -> Session:
    console = Console(file=io.StringIO(), record=True, width=100)
    provider = _FakeProvider(completions)
    agent = Agent(provider, [], system_prompt="sys", compactor=compactor)
    return Session(console=console, provider=provider, provider_name="anthropic", agent=agent)


def _completion(text: str, **usage: int) -> Completion:
    return Completion(text=text, usage=Usage(**usage))


def _text(result: CommandResult) -> str:
    assert result.message is not None
    console = Console(file=io.StringIO(), record=True, width=200)
    console.print(result.message)
    return console.export_text()


def test_unknown_command_is_friendly_and_never_runs() -> None:
    result = dispatch(_session(), "/nope")

    assert not result.exit_session
    assert "Unknown command" in _text(result)


def test_help_lists_every_command() -> None:
    text = _text(dispatch(_session(), "/help"))

    for command in registered_commands():
        assert f"/{command.name}" in text


def test_exit_signals_the_session_to_stop() -> None:
    result = dispatch(_session(), "/exit")

    assert result.exit_session
    assert "Goodbye" in _text(result)


def test_clear_resets_the_conversation() -> None:
    session = _session([_completion("ok")])
    session.run_turn("remember figs")

    result = dispatch(session, "/clear")

    assert "cleared" in _text(result).lower()


def test_usage_without_a_turn_shows_zero_tokens_and_no_context() -> None:
    text = _text(dispatch(_session(), "/usage"))

    assert "↑ 0 ↓ 0 tokens" in text
    assert "ctx" not in text  # no context read before the first turn


def test_usage_after_a_turn_shows_tokens_and_context() -> None:
    session = _session([_completion("hi", input_tokens=12000, output_tokens=500)])
    session.run_turn("go")

    text = _text(dispatch(session, "/usage"))

    assert "↑ 12.0k ↓ 500 tokens" in text
    assert "ctx" in text  # the context read appears once a turn has run


def test_model_shows_provider_model_and_effort() -> None:
    text = _text(dispatch(_session(), "/model"))

    assert "anthropic" in text
    assert "fake-model" in text
    assert "effort off" in text


def test_model_with_an_argument_explains_switching_is_unsupported() -> None:
    text = _text(dispatch(_session(), "/model gpt-4o"))

    assert "isn't supported" in text


def test_effort_without_an_argument_shows_the_current_level() -> None:
    text = _text(dispatch(_session(), "/effort"))

    assert "off" in text


def test_effort_sets_a_valid_level() -> None:
    session = _session()

    result = dispatch(session, "/effort high")

    assert session.effort is ReasoningEffort.HIGH
    assert "high" in _text(result)


def test_effort_rejects_an_unknown_level() -> None:
    session = _session()

    result = dispatch(session, "/effort turbo")

    assert session.effort is ReasoningEffort.OFF  # left unchanged
    assert "Unknown effort" in _text(result)


def test_compact_reports_nothing_to_do_for_a_short_conversation() -> None:
    text = _text(dispatch(_session(), "/compact"))  # default no-op compactor

    assert "Nothing to compact" in text


def test_compact_summarises_and_reports_what_changed() -> None:
    session = _session(
        [_completion("a1"), _completion("a2"), _completion("a3")],
        compactor=SummarisingCompactor(keep_recent_turns=1),
    )
    session.run_turn("u1")
    session.run_turn("u2")
    session.run_turn("u3")

    text = _text(dispatch(session, "/compact"))

    assert "Compacted" in text  # the recap replaced the older turns

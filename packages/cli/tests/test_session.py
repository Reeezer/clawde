from __future__ import annotations

import io
from collections.abc import Iterator, Sequence

import pytest
from clawde_core.loop import Agent
from clawde_core.models import (
    Completion,
    Message,
    ReasoningEffort,
    StreamChunk,
    ToolSpec,
    Usage,
)
from clawde_core.providers.base import ModelProvider, ProviderError
from rich.console import Console

from clawde_cli import session as session_module
from clawde_cli.session import (
    Session,
    TurnStatus,
    _derive_title,
    build_session,
    default_system_prompt,
)


class _FakeProvider(ModelProvider):
    """Streams a scripted reply (or raises) so a real Agent can drive a turn."""

    def __init__(
        self,
        completions: Sequence[Completion] = (),
        *,
        error: BaseException | None = None,
        context_window: int = 1000,
    ) -> None:
        self._queue = list(completions)
        self._error = error
        self._window = context_window
        self.received: list[tuple[Message, ...]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    @property
    def context_window(self) -> int:
        return self._window

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        raise NotImplementedError  # the Session drives via stream()

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        self.received.append(tuple(messages))
        if self._error is not None:
            raise self._error
        completion = self._queue.pop(0)
        if completion.text:
            yield StreamChunk(text=completion.text)
        yield StreamChunk(completion=completion)


class _NoCompletionProvider(ModelProvider):
    """Streams a delta but never a completion, so the loop raises AgentError."""

    @property
    def model(self) -> str:
        return "fake-model"

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        raise NotImplementedError

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        yield StreamChunk(text="partial")


def _recording_console() -> Console:
    return Console(file=io.StringIO(), record=True, width=80)


def _session(provider: ModelProvider, console: Console | None = None) -> Session:
    console = console if console is not None else _recording_console()
    agent = Agent(provider, [], system_prompt="sys")
    return Session(console=console, provider=provider, provider_name="fake", agent=agent)


def _completion(text: str, **usage: int) -> Completion:
    return Completion(text=text, usage=Usage(**usage))


# --- run_turn outcomes --------------------------------------------------------


def test_run_turn_streams_the_reply_and_returns_ok() -> None:
    provider = _FakeProvider([_completion("here are the files", input_tokens=3, output_tokens=4)])
    console = _recording_console()
    session = _session(provider, console)

    outcome = session.run_turn("list files")

    assert outcome.status is TurnStatus.OK
    assert outcome.ok
    out = console.export_text()
    assert "here are the files" in out  # streamed reply
    assert "↑ 3 ↓ 4 tokens" in out  # closing summary


def test_run_turn_reports_a_provider_error() -> None:
    provider = _FakeProvider(error=ProviderError("api key missing"))
    console = _recording_console()
    session = _session(provider, console)

    outcome = session.run_turn("hi")

    assert outcome.status is TurnStatus.ERROR
    assert not outcome.ok
    assert outcome.message == "api key missing"
    assert "api key missing" in console.export_text()


def test_run_turn_reports_an_agent_error() -> None:
    console = _recording_console()
    session = _session(_NoCompletionProvider(), console)

    outcome = session.run_turn("hi")

    assert outcome.status is TurnStatus.ERROR
    assert "stream ended" in (outcome.message or "")


def test_run_turn_handles_a_keyboard_interrupt() -> None:
    provider = _FakeProvider(error=KeyboardInterrupt())
    console = _recording_console()
    session = _session(provider, console)

    outcome = session.run_turn("hi")

    assert outcome.status is TurnStatus.INTERRUPTED
    assert "Interrupted" in console.export_text()


# --- running totals -----------------------------------------------------------


def test_usage_accumulates_across_turns() -> None:
    provider = _FakeProvider(
        [
            _completion("one", input_tokens=10, output_tokens=2),
            _completion("two", input_tokens=20, output_tokens=3),
        ]
    )
    session = _session(provider)

    session.run_turn("first")
    session.run_turn("second")

    assert session.usage == Usage(input_tokens=30, output_tokens=5)


def test_a_failed_turn_leaves_usage_unchanged() -> None:
    session = _session(_FakeProvider(error=ProviderError("nope")))

    session.run_turn("hi")

    assert session.usage == Usage()


def test_title_is_derived_from_the_first_prompt_only() -> None:
    provider = _FakeProvider([_completion("a"), _completion("b")])
    session = _session(provider)
    assert session.title is None

    session.run_turn("add a REPL")
    session.run_turn("now add tests")

    assert session.title == "add a REPL"


def test_context_budget_is_none_until_a_turn_runs() -> None:
    provider = _FakeProvider([_completion("a", input_tokens=5, output_tokens=1)])
    session = _session(provider)
    assert session.context_budget is None

    session.run_turn("hi")

    budget = session.context_budget
    assert budget is not None
    assert budget.limit == 1000  # the provider's context window


# --- commands act on the session ---------------------------------------------


def test_clear_resets_the_conversation() -> None:
    provider = _FakeProvider([_completion("a"), _completion("b")])
    session = _session(provider)
    session.run_turn("remember pears")
    session.clear()

    session.run_turn("what now")

    second_sent = provider.received[-1]
    assert all("pears" not in message.content for message in second_sent)


def test_set_effort_changes_the_provider_effort() -> None:
    session = _session(_FakeProvider([_completion("a")]))

    start = session.effort
    session.set_effort(ReasoningEffort.HIGH)
    end = session.effort

    assert start is ReasoningEffort.OFF
    assert end is ReasoningEffort.HIGH


def test_model_and_provider_name_are_exposed() -> None:
    session = _session(_FakeProvider([_completion("a")]))

    assert session.model == "fake-model"
    assert session.provider_name == "fake"


# --- build_session + helpers --------------------------------------------------


def test_build_session_resolves_provider_name_and_applies_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(session_module, "build_provider", lambda p=None, m=None: fake)
    monkeypatch.setattr(session_module, "build_tools", lambda settings: [])

    session = build_session(_recording_console(), provider="anthropic", effort=ReasoningEffort.HIGH)

    assert session.provider_name == "anthropic"
    assert session.effort is ReasoningEffort.HIGH


def test_build_session_defaults_provider_name_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(session_module, "build_provider", lambda p=None, m=None: fake)
    monkeypatch.setattr(session_module, "build_tools", lambda settings: [])

    session = build_session(_recording_console())

    assert session.provider_name  # the settings default, never empty
    assert session.effort is ReasoningEffort.OFF  # no override -> provider default


def test_default_system_prompt_names_clawde_and_its_tools() -> None:
    prompt = default_system_prompt()

    assert "clawde" in prompt
    assert "bash" in prompt


def test_derive_title_trims_truncates_and_handles_blank() -> None:
    assert _derive_title("  hello world  ") == "hello world"
    assert _derive_title("   ") == ""

    title = _derive_title("x" * 80)

    assert title.endswith("…")
    assert len(title) == 50

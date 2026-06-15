from __future__ import annotations

import io
from collections.abc import Callable, Iterator, Sequence

import pytest
from clawde_core.loop import Agent
from clawde_core.models import Completion, Message, StreamChunk, ToolSpec, Usage
from clawde_core.providers.base import ModelProvider
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from clawde_cli import repl
from clawde_cli.gitinfo import GitContext
from clawde_cli.repl import interactive_line_reader, run_repl
from clawde_cli.session import Session


class _FakeProvider(ModelProvider):
    def __init__(self, completions: Sequence[Completion] = ()) -> None:
        self._queue = list(completions)
        self.received: list[tuple[Message, ...]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    @property
    def context_window(self) -> int:
        return 1000

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        raise NotImplementedError

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        self.received.append(tuple(messages))
        completion = self._queue.pop(0)
        if completion.text:
            yield StreamChunk(text=completion.text)
        yield StreamChunk(completion=completion)


def _scripted_reader(*steps: str | BaseException) -> Callable[[], str]:
    """A line reader that replays ``steps`` (a string is returned, an exception
    raised), then raises EOFError — like Ctrl-D at the end of input."""
    iterator = iter(steps)

    def read() -> str:
        try:
            step = next(iterator)
        except StopIteration:
            raise EOFError from None
        if isinstance(step, BaseException):
            raise step
        return step

    return read


def _console() -> Console:
    return Console(file=io.StringIO(), record=True, width=100)


def _session(provider: ModelProvider, console: Console) -> Session:
    agent = Agent(provider, [], system_prompt="sys")
    return Session(console=console, provider=provider, provider_name="anthropic", agent=agent)


@pytest.fixture(autouse=True)
def _stub_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repl, "current_git_context", lambda: GitContext(branch="feature/x", worktree="wt")
    )


def test_header_and_hint_are_printed_then_eof_ends_the_loop() -> None:
    console = _console()
    provider = _FakeProvider()

    run_repl(_session(provider, console), console, _scripted_reader())

    out = console.export_text()
    assert "feature/x" in out  # the header announces the branch
    assert "fake-model" in out  # ... and the active model
    assert "/help" in out  # the hint points at the commands
    assert provider.received == []  # no input -> no turn


def test_a_plain_line_runs_a_turn() -> None:
    console = _console()
    provider = _FakeProvider([Completion(text="hi there", usage=Usage(output_tokens=2))])

    run_repl(_session(provider, console), console, _scripted_reader("say hi"))

    assert len(provider.received) == 1  # the prompt drove one turn
    assert "hi there" in console.export_text()


def test_blank_lines_are_ignored() -> None:
    console = _console()
    provider = _FakeProvider()

    run_repl(_session(provider, console), console, _scripted_reader("", "   "))

    assert provider.received == []  # nothing reached the model


def test_a_slash_command_is_dispatched_not_sent_to_the_model() -> None:
    console = _console()
    provider = _FakeProvider()

    run_repl(_session(provider, console), console, _scripted_reader("/help"))

    assert provider.received == []  # the command never reached the model
    assert "/clear" in console.export_text()  # /help listed the commands


def test_exit_command_stops_before_reading_more() -> None:
    console = _console()
    provider = _FakeProvider([Completion(text="unused")])

    run_repl(_session(provider, console), console, _scripted_reader("/exit", "should not run"))

    assert provider.received == []  # the loop broke at /exit, never read the next line
    assert "Goodbye" in console.export_text()


def test_ctrl_c_at_the_prompt_cancels_the_line_and_continues() -> None:
    console = _console()
    provider = _FakeProvider()

    run_repl(_session(provider, console), console, _scripted_reader(KeyboardInterrupt(), "/exit"))

    assert "Goodbye" in console.export_text()  # survived the interrupt, then exited cleanly


def test_interactive_line_reader_reads_a_line() -> None:
    with create_pipe_input() as pipe:
        pipe.send_text("hello\r")
        reader = interactive_line_reader(pt_input=pipe, output=DummyOutput())

        assert reader() == "hello"

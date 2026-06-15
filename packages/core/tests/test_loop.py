from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

import pytest

from clawde_core.loop import Agent, AgentError
from clawde_core.models import (
    Completion,
    ImageContent,
    Message,
    Role,
    StreamChunk,
    ThinkingBlock,
    ToolCall,
    ToolSpec,
    Usage,
)
from clawde_core.providers.base import ModelProvider
from clawde_core.tools.base import Tool


class FakeProvider(ModelProvider):
    """Returns queued completions in order; records the messages it was sent.

    Has no ``stream()`` override, so streaming tests exercise the base default.
    """

    def __init__(self, completions: Sequence[Completion]) -> None:
        self._queue = list(completions)
        self.received: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        self.received.append(tuple(messages))
        return self._queue.pop(0)


class FakeStreamingProvider(ModelProvider):
    """Yields scripted stream chunks, one script per model call."""

    def __init__(self, scripts: Sequence[Sequence[StreamChunk]]) -> None:
        self._scripts = [list(script) for script in scripts]

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        raise NotImplementedError  # streaming tests exercise stream() only

    def stream(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> Iterator[StreamChunk]:
        yield from self._scripts.pop(0)


class RecordingTool(Tool):
    """A no-op tool that records the arguments it was called with."""

    def __init__(self, name: str = "echo", output: str = "tool-output") -> None:
        self._name = name
        self._output = output
        self.calls: list[Mapping[str, object]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name=self._name, description="a recording tool", parameters={})

    def run(self, arguments: Mapping[str, object]) -> str:
        self.calls.append(dict(arguments))
        return self._output


def test_answers_without_tools() -> None:
    provider = FakeProvider(
        [Completion(text="hello", usage=Usage(input_tokens=5, output_tokens=2))]
    )
    agent = Agent(provider, [], system_prompt="sys")

    turn = agent.run_turn("hi")

    assert turn.final_text == "hello"
    assert turn.usage == Usage(input_tokens=5, output_tokens=2)
    assert provider.received[0][0] == Message.system("sys")
    assert provider.received[0][1] == Message.user("hi")


def test_thinking_blocks_are_carried_onto_the_assistant_turn() -> None:
    block = ThinkingBlock(text="reasoned", signature="sig-1")
    provider = FakeProvider([Completion(text="answer", thinking=(block,))])
    agent = Agent(provider, [], system_prompt="s")

    turn = agent.run_turn("hi")

    assert turn.messages[1].thinking == (block,)  # preserved for the next request


def test_run_turn_attaches_images_to_the_user_message() -> None:
    image = ImageContent(mime_type="image/png", data=b"png")
    provider = FakeProvider([Completion(text="a diagram")])
    agent = Agent(provider, [], system_prompt="s")

    agent.run_turn("describe this", images=(image,))

    user_message = provider.received[0][1]
    assert user_message.images == (image,)


def test_executes_tool_then_answers() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"x": 1})
    provider = FakeProvider(
        [
            Completion(tool_calls=(call,), usage=Usage(output_tokens=1)),
            Completion(text="done", usage=Usage(output_tokens=1)),
        ]
    )
    tool = RecordingTool(name="echo", output="ran")
    agent = Agent(provider, [tool], system_prompt="sys")

    turn = agent.run_turn("do it")

    assert turn.final_text == "done"
    assert tool.calls == [{"x": 1}]
    assert turn.usage == Usage(output_tokens=2)  # summed across both model calls
    fed_back = provider.received[1]
    assert any(
        m.role is Role.TOOL and m.content == "ran" and m.name == "echo" and m.tool_call_id == "c1"
        for m in fed_back
    )


def test_executes_multiple_tool_calls_in_one_step() -> None:
    calls = (ToolCall(id="a", name="echo"), ToolCall(id="b", name="echo"))
    provider = FakeProvider([Completion(tool_calls=calls), Completion(text="ok")])
    tool = RecordingTool(name="echo")
    agent = Agent(provider, [tool], system_prompt="s")

    turn = agent.run_turn("go")

    assert len(tool.calls) == 2
    assert turn.final_text == "ok"


def test_unknown_tool_is_reported_back_not_raised() -> None:
    call = ToolCall(id="c1", name="ghost")
    provider = FakeProvider([Completion(tool_calls=(call,)), Completion(text="sorry")])
    agent = Agent(provider, [RecordingTool(name="echo")], system_prompt="s")

    turn = agent.run_turn("use ghost")

    fed_back = provider.received[1]
    tool_msg = next(m for m in fed_back if m.role is Role.TOOL)
    assert "unknown tool 'ghost'" in tool_msg.content
    assert turn.final_text == "sorry"


def test_raises_when_exceeding_iteration_budget() -> None:
    call = ToolCall(id="c1", name="echo")
    provider = FakeProvider([Completion(tool_calls=(call,))] * 3)
    agent = Agent(provider, [RecordingTool(name="echo")], system_prompt="s", max_iterations=3)

    with pytest.raises(AgentError, match="within 3 iterations"):
        agent.run_turn("loop forever")


def test_history_persists_but_turn_carries_only_its_messages() -> None:
    provider = FakeProvider([Completion(text="one"), Completion(text="two")])
    agent = Agent(provider, [], system_prompt="s")

    first = agent.run_turn("a")
    second = agent.run_turn("b")

    assert [m.content for m in first.messages] == ["a", "one"]
    assert [m.content for m in second.messages] == ["b", "two"]
    # the second call saw the full prior history (system + 2 turns of context)
    assert len(provider.received[1]) == 1 + 3  # system + (user a, asst one, user b)


def test_stream_turn_falls_back_to_default_stream() -> None:
    # FakeProvider has no stream() override, so the ABC default (one terminal
    # chunk wrapping complete()) is used: no incremental text, same final answer.
    provider = FakeProvider([Completion(text="hello", usage=Usage(output_tokens=2))])
    agent = Agent(provider, [], system_prompt="s")
    seen: list[str] = []

    turn = agent.stream_turn("hi", on_text=seen.append)

    assert seen == []
    assert turn.final_text == "hello"
    assert turn.usage == Usage(output_tokens=2)


def test_stream_turn_streams_text_deltas() -> None:
    provider = FakeStreamingProvider(
        [
            [
                StreamChunk(text="Hel"),
                StreamChunk(text="lo"),
                StreamChunk(completion=Completion(text="Hello", usage=Usage(output_tokens=1))),
            ]
        ]
    )
    agent = Agent(provider, [], system_prompt="s")
    seen: list[str] = []

    turn = agent.stream_turn("hi", on_text=seen.append)

    assert seen == ["Hel", "lo"]
    assert turn.final_text == "Hello"


def test_stream_turn_announces_tool_calls() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"x": 1})
    provider = FakeStreamingProvider(
        [
            [StreamChunk(completion=Completion(tool_calls=(call,)))],
            [StreamChunk(text="done"), StreamChunk(completion=Completion(text="done"))],
        ]
    )
    tool = RecordingTool(name="echo", output="ran")
    agent = Agent(provider, [tool], system_prompt="s")
    announced: list[str] = []

    turn = agent.stream_turn(
        "go", on_text=lambda _delta: None, on_tool_call=lambda c: announced.append(c.name)
    )

    assert announced == ["echo"]
    assert tool.calls == [{"x": 1}]
    assert turn.final_text == "done"


def test_stream_turn_surfaces_tool_results() -> None:
    call = ToolCall(id="c1", name="echo", arguments={"x": 1})
    provider = FakeStreamingProvider(
        [
            [StreamChunk(completion=Completion(tool_calls=(call,)))],
            [StreamChunk(text="done"), StreamChunk(completion=Completion(text="done"))],
        ]
    )
    tool = RecordingTool(name="echo", output="ran")
    agent = Agent(provider, [tool], system_prompt="s")
    results: list[tuple[str, str]] = []

    turn = agent.stream_turn(
        "go",
        on_text=lambda _delta: None,
        on_tool_result=lambda r: results.append((r.tool_call_id, r.content)),
    )

    assert results == [("c1", "ran")]
    assert turn.final_text == "done"


def test_stream_turn_reports_running_usage_after_each_model_call() -> None:
    call = ToolCall(id="c1", name="echo")
    provider = FakeStreamingProvider(
        [
            [StreamChunk(completion=Completion(tool_calls=(call,), usage=Usage(output_tokens=3)))],
            [StreamChunk(completion=Completion(text="done", usage=Usage(output_tokens=4)))],
        ]
    )
    agent = Agent(provider, [RecordingTool(name="echo")], system_prompt="s")
    seen: list[Usage] = []

    agent.stream_turn("go", on_text=lambda _delta: None, on_usage=seen.append)

    # One report per model call, each carrying the cumulative total so far.
    assert [u.output_tokens for u in seen] == [3, 7]


def test_stream_turn_raises_if_stream_has_no_completion() -> None:
    provider = FakeStreamingProvider([[StreamChunk(text="partial")]])
    agent = Agent(provider, [], system_prompt="s")

    with pytest.raises(AgentError, match="without a completion"):
        agent.stream_turn("hi", on_text=lambda _delta: None)

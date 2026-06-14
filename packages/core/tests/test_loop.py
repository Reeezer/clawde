from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from clawde_core.loop import Agent, AgentError
from clawde_core.models import (
    Completion,
    Message,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from clawde_core.providers.base import ModelProvider
from clawde_core.tools.base import Tool


class FakeProvider(ModelProvider):
    """Returns queued completions in order; records the messages it was sent."""

    def __init__(self, completions: Sequence[Completion]) -> None:
        self._queue = list(completions)
        self.received: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        self.received.append(tuple(messages))
        return self._queue.pop(0)


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

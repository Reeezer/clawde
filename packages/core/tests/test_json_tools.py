from __future__ import annotations

import json
from collections.abc import Sequence

from clawde_core.models import (
    Completion,
    ImageContent,
    Message,
    ReasoningEffort,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from clawde_core.providers import json_tools
from clawde_core.providers.base import ModelProvider
from clawde_core.providers.json_tools import JsonToolCallingProvider

_SPEC = ToolSpec(
    name="bash",
    description="run a shell command",
    parameters={"type": "object", "properties": {"command": {"type": "string"}}},
)


class _Inner(ModelProvider):
    """Fake inner provider: records what it was sent, returns a scripted reply."""

    def __init__(self, reply: Completion) -> None:
        self._reply = reply
        self.seen: tuple[Message, ...] = ()
        self.saw_tools: tuple[ToolSpec, ...] = ()

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        self.seen = tuple(messages)
        self.saw_tools = tuple(tools)
        return self._reply


def _call_block(name: str, arguments: object) -> str:
    return f"```json\n{json.dumps({'tool': name, 'arguments': arguments})}\n```"


# --- complete(): parsing + delegation ----------------------------------------


def test_complete_parses_a_fenced_tool_call() -> None:
    reply = Completion(
        text="let me look\n" + _call_block("bash", {"command": "ls"}),
        usage=Usage(output_tokens=3),
    )
    inner = _Inner(reply)

    completion = JsonToolCallingProvider(inner).complete(
        [Message.system("be helpful"), Message.user("list files")], [_SPEC]
    )

    assert [c.name for c in completion.tool_calls] == ["bash"]
    assert completion.tool_calls[0].arguments == {"command": "ls"}
    assert completion.text == "let me look"
    assert completion.usage == Usage(output_tokens=3)
    assert inner.saw_tools == ()  # native tools suppressed; the convention lives in the prompt


def test_complete_injects_protocol_and_tools_into_system_prompt() -> None:
    inner = _Inner(Completion(text="done"))

    JsonToolCallingProvider(inner).complete(
        [Message.system("be helpful"), Message.user("hi")], [_SPEC]
    )

    system = inner.seen[0]
    assert system.role is Role.SYSTEM
    assert "be helpful" in system.content  # original system prompt preserved
    assert "bash" in system.content  # the tool is advertised
    assert "json" in system.content.lower()  # the convention is described


def test_complete_without_tools_delegates_and_skips_parsing() -> None:
    # The reply contains a json block, but no tools were offered -> not parsed as a call.
    reply = Completion(text=_call_block("bash", {"command": "ls"}), usage=Usage(output_tokens=1))
    inner = _Inner(reply)

    completion = JsonToolCallingProvider(inner).complete([Message.user("hi")], [])

    assert completion.tool_calls == ()
    assert completion is reply  # delegated unchanged


def test_complete_inserts_a_system_prompt_when_none_present() -> None:
    inner = _Inner(Completion(text="ok"))

    JsonToolCallingProvider(inner).complete([Message.user("hi")], [_SPEC])

    assert inner.seen[0].role is Role.SYSTEM  # protocol prepended as a fresh system turn
    assert inner.seen[1].role is Role.USER


def test_stream_uses_the_base_default_and_still_parses_calls() -> None:
    reply = Completion(text=_call_block("bash", {"command": "ls"}), usage=Usage(output_tokens=2))
    provider = JsonToolCallingProvider(_Inner(reply))

    chunks = list(provider.stream([Message.system("s"), Message.user("go")], [_SPEC]))

    final = chunks[-1].completion
    assert final is not None
    assert [c.name for c in final.tool_calls] == ["bash"]


# --- context window & token counting delegation ------------------------------


class _WindowingInner(ModelProvider):
    """Inner provider with a distinctive window + a recording token counter."""

    def __init__(self) -> None:
        self.counted: tuple[tuple[Message, ...], tuple[ToolSpec, ...]] | None = None

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        return Completion()

    @property
    def context_window(self) -> int:
        return 4242

    def count_tokens(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> int:
        self.counted = (tuple(messages), tuple(tools))
        return 77


def test_context_window_delegates_to_the_inner_provider() -> None:
    assert JsonToolCallingProvider(_WindowingInner()).context_window == 4242


def test_count_tokens_counts_the_shaped_conversation_with_no_native_tools() -> None:
    inner = _WindowingInner()

    counted = JsonToolCallingProvider(inner).count_tokens(
        [Message.system("sys"), Message.user("hi")], [_SPEC]
    )

    assert counted == 77
    assert inner.counted is not None
    shaped_messages, shaped_tools = inner.counted
    assert shaped_tools == ()  # native tools suppressed; the convention lives in the prompt
    assert "bash" in shaped_messages[0].content  # the shaped system prompt advertises the tool


def test_reasoning_effort_delegates_to_the_inner_provider() -> None:
    inner = _Inner(Completion(text="ok"))
    wrapper = JsonToolCallingProvider(inner)

    assert wrapper.reasoning_effort is ReasoningEffort.OFF
    wrapper.set_reasoning_effort(ReasoningEffort.HIGH)

    assert inner.reasoning_effort is ReasoningEffort.HIGH  # the inner provider builds the request
    mirrored: ReasoningEffort = wrapper.reasoning_effort
    assert mirrored is ReasoningEffort.HIGH  # and the wrapper mirrors it


# --- _shape(): flattening tool calls / results / images ----------------------


def test_shape_flattens_assistant_tool_calls_and_tool_results_to_text() -> None:
    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    conversation = [
        Message.system("sys"),
        Message.user("list files"),
        Message.assistant(content="on it", tool_calls=(call,)),
        Message.tool(tool_call_id="c1", content="a.py b.py", name="bash"),
    ]

    shaped = json_tools._shape(conversation, [_SPEC])

    assistant = shaped[2]
    assert assistant.role is Role.ASSISTANT
    assert assistant.tool_calls == ()  # the call is now plain text, not a native call
    assert "bash" in assistant.content
    assert "on it" in assistant.content
    tool_result = shaped[3]
    assert tool_result.role is Role.USER  # the model has no native tool role
    assert "a.py b.py" in tool_result.content


def test_shape_preserves_user_images() -> None:
    image = ImageContent(mime_type="image/png", data=b"png")

    shaped = json_tools._shape([Message.user("what is this?", images=(image,))], [_SPEC])

    assert shaped[-1].images == (image,)  # the user turn keeps its image


def test_shape_without_tools_leaves_the_system_prompt_untouched() -> None:
    shaped = json_tools._shape([Message.system("sys"), Message.user("hi")], [])

    assert shaped[0].content == "sys"  # no protocol appended when no tools are offered


# --- _parse(): the tool-call extractor ---------------------------------------


def test_parse_extracts_multiple_calls_and_strips_them() -> None:
    text = (
        "first\n"
        + _call_block("bash", {"command": "ls"})
        + "\nthen\n"
        + _call_block("bash", {"command": "pwd"})
    )

    clean, calls = json_tools._parse(text)

    assert [c.arguments for c in calls] == [{"command": "ls"}, {"command": "pwd"}]
    assert "```" not in clean
    assert "first" in clean
    assert "then" in clean


def test_parse_accepts_a_bare_json_object_without_a_fence() -> None:
    clean, calls = json_tools._parse(json.dumps({"tool": "bash", "arguments": {"command": "ls"}}))

    assert clean == ""
    assert [c.name for c in calls] == ["bash"]


def test_parse_returns_plain_text_when_no_call_present() -> None:
    clean, calls = json_tools._parse("just a normal answer")

    assert clean == "just a normal answer"
    assert calls == ()


def test_parse_ignores_malformed_non_dict_and_keyless_json() -> None:
    text = '```json\nnot valid json\n```\n```json\n123\n```\n```json\n{"no": "tool key"}\n```'

    clean, calls = json_tools._parse(text)

    assert calls == ()  # malformed, non-dict, and tool-less blocks are all skipped
    assert "tool key" in clean  # nothing recognised -> original text returned


def test_parse_defaults_non_dict_arguments_to_empty() -> None:
    _clean, calls = json_tools._parse(_call_block("bash", "oops"))

    assert calls[0].arguments == {}

from __future__ import annotations

import pytest
from pydantic import ValidationError

from clawde_core.models import (
    Completion,
    ImageContent,
    Message,
    Role,
    StreamChunk,
    TokenBudget,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)


def test_role_values() -> None:
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.TOOL.value == "tool"


def test_message_defaults() -> None:
    msg = Message(role=Role.USER, content="hi")
    assert msg.role is Role.USER
    assert msg.content == "hi"
    assert msg.tool_calls == ()
    assert msg.tool_call_id is None
    assert msg.name is None


def test_message_is_frozen() -> None:
    msg = Message(role=Role.USER, content="hi")
    with pytest.raises(ValidationError):
        msg.content = "changed"


def test_message_constructors() -> None:
    assert Message.system("rules").role is Role.SYSTEM
    assert Message.user("hello").content == "hello"

    call = ToolCall(id="c1", name="bash", arguments={"command": "ls"})
    assistant = Message.assistant(content="thinking", tool_calls=(call,))
    assert assistant.role is Role.ASSISTANT
    assert assistant.tool_calls == (call,)

    result = Message.tool(tool_call_id="c1", content="out", name="bash")
    assert result.role is Role.TOOL
    assert result.tool_call_id == "c1"
    assert result.name == "bash"
    assert result.content == "out"


def test_image_content_is_frozen_value_object() -> None:
    image = ImageContent(mime_type="image/png", data=b"\x89PNGdata")
    assert image.mime_type == "image/png"
    assert image.data == b"\x89PNGdata"
    with pytest.raises(ValidationError):
        image.data = b"changed"


def test_message_carries_images_and_defaults_to_none() -> None:
    assert Message.user("hi").images == ()  # additive: the str path is untouched

    image = ImageContent(mime_type="image/jpeg", data=b"jpg")
    msg = Message.user("what is this?", images=(image,))
    assert msg.content == "what is this?"
    assert msg.images == (image,)


def test_tool_call_defaults_to_empty_arguments() -> None:
    call = ToolCall(id="c1", name="bash")
    assert call.arguments == {}


def test_tool_result() -> None:
    result = ToolResult(tool_call_id="c1", content="output")
    assert result.tool_call_id == "c1"
    assert result.content == "output"


def test_usage_total_and_defaults() -> None:
    usage = Usage()
    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (0, 0, 0)
    assert usage.total == 0

    usage = Usage(input_tokens=10, output_tokens=3, cached_tokens=2)
    assert usage.total == 13


def test_usage_addition_is_immutable() -> None:
    a = Usage(input_tokens=10, output_tokens=3, cached_tokens=1)
    b = Usage(input_tokens=5, output_tokens=7, cached_tokens=4)
    combined = a + b
    assert combined == Usage(input_tokens=15, output_tokens=10, cached_tokens=5)
    # operands are untouched (immutability)
    assert a.input_tokens == 10
    assert b.output_tokens == 7


def test_completion_defaults() -> None:
    completion = Completion()
    assert completion.text == ""
    assert completion.tool_calls == ()
    assert completion.usage == Usage()


def test_stream_chunk() -> None:
    delta = StreamChunk(text="hi")
    assert delta.text == "hi"
    assert delta.completion is None

    terminal = StreamChunk(completion=Completion(text="done"))
    assert terminal.text == ""
    assert terminal.completion is not None
    assert terminal.completion.text == "done"


def test_tool_spec() -> None:
    spec = ToolSpec(
        name="bash",
        description="run a command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    assert spec.name == "bash"
    assert spec.parameters["type"] == "object"
    spec_default = ToolSpec(name="noop", description="d")
    assert spec_default.parameters == {}


def test_token_budget_remaining_is_window_minus_used() -> None:
    assert TokenBudget(limit=1000, used=300).remaining == 700


def test_token_budget_remaining_never_goes_negative() -> None:
    assert TokenBudget(limit=1000, used=1500).remaining == 0


def test_token_budget_fraction_is_share_of_window() -> None:
    assert TokenBudget(limit=1000, used=250).fraction == 0.25


def test_token_budget_fraction_exceeds_one_when_over_budget() -> None:
    assert TokenBudget(limit=1000, used=1200).fraction == 1.2


def test_token_budget_fraction_is_full_when_window_is_zero() -> None:
    assert TokenBudget(limit=0, used=0).fraction == 1.0

from __future__ import annotations

import io

from clawde_core.models import CompactionEvent, TokenBudget, ToolCall, ToolResult, Usage
from rich.color import Color
from rich.console import Console, RenderableType
from rich.segment import Segment
from rich.text import Text

from clawde_cli.rendering import (
    BOLD_STYLE,
    CLAWDE_THEME,
    CODE_BLOCK_STYLE,
    INLINE_CODE_STYLE,
    LIST_BULLET,
    STEP_FAIL_STYLE,
    STEP_GLYPH,
    STEP_OK_STYLE,
    ClawdeMarkdown,
    ReplyRenderer,
    format_tool_call,
    format_tool_result,
    make_console,
    result_point_style,
    step_block,
)
from clawde_cli.status import SPINNER_FRAMES, StatusReporter


def _render(markup: str) -> list[Segment]:
    console = Console(theme=CLAWDE_THEME, force_terminal=True, width=80, color_system="truecolor")
    return list(console.render(ClawdeMarkdown(markup)))


def _segment_for(markup: str, needle: str) -> Segment:
    for segment in _render(markup):
        if needle in segment.text:
            return segment
    raise AssertionError(f"no rendered segment contained {needle!r}")


def _markdown_text(markup: str, width: int = 80) -> str:
    console = Console(theme=CLAWDE_THEME, record=True, width=width)
    console.print(ClawdeMarkdown(markup))
    return console.export_text()


def _recording_console() -> Console:
    # StringIO-backed so transient glyphs never hit a codepage; non-terminal, so
    # the Live region renders only at stop — i.e. correct-at-end.
    return Console(theme=CLAWDE_THEME, file=io.StringIO(), record=True, width=80)


def _capture(renderable: RenderableType) -> str:
    console = Console(theme=CLAWDE_THEME, file=io.StringIO(), record=True, width=80)
    console.print(renderable)
    return console.export_text()


def _stub_status() -> StatusReporter:
    return StatusReporter(clock=lambda: 0.0, choose=lambda verbs: verbs[0])


def test_inline_code_is_pastel_purple() -> None:
    segment = _segment_for("Run the `loop` now.", "loop")
    assert segment.style is not None
    assert segment.style.color == Color.parse(INLINE_CODE_STYLE)


def test_fenced_block_is_flat_pastel_green() -> None:
    segment = _segment_for("```\nprint(1)\n```", "print(1)")
    assert segment.style is not None
    assert segment.style.color == Color.parse(CODE_BLOCK_STYLE)


def test_language_tagged_fence_is_still_flat_green() -> None:
    segment = _segment_for("```python\nx = 1\n```", "x = 1")
    assert segment.style is not None
    assert segment.style.color == Color.parse(CODE_BLOCK_STYLE)


def test_bold_is_bright_white() -> None:
    segment = _segment_for("This is **important** text.", "important")
    assert segment.style is not None
    assert segment.style.bold
    assert segment.style.color == Color.parse("bright_white")


def test_plain_text_has_no_colour() -> None:
    segment = _segment_for("just plain words", "plain")
    assert segment.style is None or segment.style.color is None


def test_markdown_list_uses_a_dash_bullet_not_a_dot() -> None:
    out = _markdown_text("- alpha\n- beta")
    assert f"{LIST_BULLET} alpha" in out
    assert f"{LIST_BULLET} beta" in out
    assert STEP_GLYPH not in out  # the bigger step point never appears in markdown


def test_make_console_carries_the_theme() -> None:
    console = make_console()
    assert console.get_style("markdown.code").color == Color.parse(INLINE_CODE_STYLE)
    assert console.get_style("markdown.code_block").color == Color.parse(CODE_BLOCK_STYLE)
    assert console.get_style("markdown.strong") == console.get_style(BOLD_STYLE)


def test_format_tool_call_summarises_bash_command_without_a_glyph() -> None:
    text = format_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls -la"}))
    assert text.plain == 'bash(command="ls -la")'


def test_format_tool_call_shows_only_a_tools_primary_args() -> None:
    call = ToolCall(id="1", name="read", arguments={"path": "loop.py", "offset": 5, "limit": 50})
    assert format_tool_call(call).plain == 'read(path="loop.py", limit=50)'


def test_format_tool_call_falls_back_to_all_args_for_unknown_tools() -> None:
    call = ToolCall(id="1", name="mystery", arguments={"a": 1, "b": "two"})
    assert format_tool_call(call).plain == 'mystery(a=1, b="two")'


def test_format_tool_call_truncates_long_argument_values() -> None:
    text = format_tool_call(ToolCall(id="1", name="bash", arguments={"command": "x" * 200}))
    assert "…" in text.plain
    assert len(text.plain) < 200


def test_format_tool_call_styles_name_and_args() -> None:
    text = format_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls"}))
    styles = {span.style for span in text.spans}
    assert {"tool.name", "tool.args"} <= styles


def test_format_tool_result_previews_first_line_and_flags_more() -> None:
    text = format_tool_result(ToolResult(tool_call_id="1", content="first line\nsecond line"))
    assert "first line" in text.plain
    assert "second line" not in text.plain
    assert "…" in text.plain


def test_format_tool_result_truncates_a_long_single_line() -> None:
    text = format_tool_result(ToolResult(tool_call_id="1", content="y" * 200))
    assert "…" in text.plain
    assert len(text.plain) < 200


def test_format_tool_result_handles_empty_output() -> None:
    text = format_tool_result(ToolResult(tool_call_id="1", content="   \n  "))
    assert "(no output)" in text.plain


def test_step_block_colours_its_leading_point() -> None:
    console = Console(force_terminal=True, width=40, color_system="truecolor")
    segments = list(console.render(step_block(STEP_OK_STYLE, Text("body"))))
    dot = next(seg for seg in segments if STEP_GLYPH in seg.text)
    assert dot.style is not None
    assert dot.style.color == Color.parse(STEP_OK_STYLE)


def test_result_point_style_is_green_for_success() -> None:
    assert result_point_style(ToolResult(tool_call_id="1", content="all good")) == STEP_OK_STYLE


def test_result_point_style_is_red_for_an_error() -> None:
    result = ToolResult(tool_call_id="1", content="Error: file not found")
    assert result_point_style(result) == STEP_FAIL_STYLE


def test_text_step_persists_behind_a_point() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console, status=_stub_status())

    renderer.begin()
    renderer.on_text("Hello ")
    renderer.on_text("**world**")
    renderer.finish()

    out = console.export_text()
    assert "Hello world" in out
    assert STEP_GLYPH in out  # the white step point sits beside the text
    assert not any(frame in out for frame in SPINNER_FRAMES)  # the spinner left no trace


def test_on_compaction_shows_a_compaction_step() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console, status=_stub_status())

    renderer.begin()
    renderer.on_compaction(
        CompactionEvent(messages_before=12, messages_after=4, tokens_before=900, tokens_after=200)
    )
    renderer.finish()

    assert "Compacted" in console.export_text()


def test_steps_are_separated_by_a_blank_line() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console)

    renderer.on_text("the thinking")
    renderer.on_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls"}))
    renderer.on_tool_result(ToolResult(tool_call_id="1", content="done"))
    renderer.finish()

    lines = [line.rstrip() for line in console.export_text().splitlines()]
    i_text = next(i for i, line in enumerate(lines) if "the thinking" in line)
    i_call = next(i for i, line in enumerate(lines) if "bash" in line)
    assert "" in lines[i_text + 1 : i_call]  # a blank line between the two steps


def test_tool_step_shows_call_then_result_in_order() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console)

    renderer.on_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls"}))
    renderer.on_tool_result(ToolResult(tool_call_id="1", content="file_a\nfile_b"))
    renderer.finish()

    out = console.export_text()
    assert out.index("bash") < out.index("file_a")


def test_interleaves_text_and_tool_steps_in_order() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console)

    renderer.on_text("looking around")
    renderer.on_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls"}))
    renderer.on_tool_result(ToolResult(tool_call_id="1", content="found"))
    renderer.on_text("found them")
    renderer.finish()

    out = console.export_text()
    assert out.index("looking around") < out.index("bash") < out.index("found them")


def test_tool_result_without_a_pending_call_shows_only_the_result() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console)

    renderer.on_tool_result(ToolResult(tool_call_id="1", content="orphan output"))
    renderer.finish()

    assert "orphan output" in console.export_text()


def test_text_free_tool_turn_leaves_no_spinner() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console, status=_stub_status())

    renderer.begin()
    renderer.on_tool_call(ToolCall(id="1", name="bash", arguments={"command": "ls"}))
    renderer.on_tool_result(ToolResult(tool_call_id="1", content="ok"))
    renderer.finish()

    out = console.export_text()
    assert out.index("bash") < out.index("ok")
    assert not any(frame in out for frame in SPINNER_FRAMES)


def test_on_usage_refreshes_the_live_spinner() -> None:
    console = _recording_console()
    status = _stub_status()
    renderer = ReplyRenderer(console, status=status)

    renderer.begin()  # spinner active, no text yet
    renderer.on_usage(Usage(output_tokens=2000))
    renderer.finish()

    assert "2.0k" in status.__rich__().plain


def test_on_usage_is_safe_with_no_live_region() -> None:
    renderer = ReplyRenderer(_recording_console(), status=_stub_status())
    renderer.on_usage(Usage(output_tokens=5))  # no begin(): nothing to refresh


def test_on_budget_shows_the_context_read_on_the_spinner() -> None:
    console = _recording_console()
    status = _stub_status()
    renderer = ReplyRenderer(console, status=status)

    renderer.begin()  # spinner active, no text yet
    renderer.on_budget(TokenBudget(limit=1_048_576, used=24100))
    renderer.finish()

    assert "ctx 24.1k/1M" in status.__rich__().plain


def test_on_budget_is_safe_with_no_live_region() -> None:
    renderer = ReplyRenderer(_recording_console(), status=_stub_status())
    renderer.on_budget(TokenBudget(limit=1_048_576, used=10))  # no begin(): nothing to refresh


def test_begin_seeds_the_pending_context_window() -> None:
    status = _stub_status()
    renderer = ReplyRenderer(_recording_console(), status=status)

    renderer.begin(1_048_576)  # window known upfront, before any reply
    assert "ctx –/1M" in status.__rich__().plain

    renderer.finish()


def test_on_budget_does_not_refresh_while_text_streams() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console, status=_stub_status())

    renderer.begin()
    renderer.on_text("streaming")
    renderer.on_budget(TokenBudget(limit=1_048_576, used=24100))  # buffer non-empty: spinner hidden
    renderer.finish()

    assert "streaming" in console.export_text()


def test_context_budget_exposes_the_last_seen_read() -> None:
    renderer = ReplyRenderer(_recording_console(), status=_stub_status())
    assert renderer.context_budget() is None  # before any model call

    renderer.on_budget(TokenBudget(limit=1_048_576, used=24100))
    assert renderer.context_budget() == TokenBudget(limit=1_048_576, used=24100)


def test_on_usage_does_not_refresh_while_text_streams() -> None:
    console = _recording_console()
    renderer = ReplyRenderer(console, status=_stub_status())

    renderer.begin()
    renderer.on_text("streaming")
    renderer.on_usage(Usage(output_tokens=9))  # buffer non-empty: spinner not shown
    renderer.finish()

    assert "streaming" in console.export_text()


def test_renderer_reports_elapsed_from_its_status() -> None:
    now = [100.0]
    status = StatusReporter(clock=lambda: now[0])
    status.start()
    now[0] = 142.0
    renderer = ReplyRenderer(_recording_console(), status=status)

    assert renderer.elapsed() == 42.0


def test_a_launching_tool_call_is_shown_live_before_its_result() -> None:
    renderer = ReplyRenderer(_recording_console(), status=_stub_status())

    renderer.on_tool_call(ToolCall(id="1", name="glob", arguments={"pattern": "**/*.py"}))
    live = renderer._live  # the call is on screen, live, while it runs
    assert live is not None
    assert "glob" in _capture(live.renderable)

    renderer.finish()

"""Render the agent's streamed reply with clawde's colour theme.

All presentation lives here in the CLI package; the engine stays UI-free
(``CLAUDE.md``). The reply is a sequence of *steps* — a block of streamed text,
or a tool call with its result — each shown with a leading point in a gutter:
white while the model writes/thinks, green when a command succeeds, red when it
fails. Steps are separated by a blank line. There is only ever one
:class:`rich.live.Live` region: it shows the working spinner (transient) or the
streaming markdown (persisted), never both — they never compete (#7/#9).
"""

from __future__ import annotations

from typing import ClassVar

from clawde_core.models import ToolCall, ToolResult, Usage
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, ListItem, Markdown, MarkdownElement
from rich.segment import Segment
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from clawde_cli.status import REFRESH_PER_SECOND, StatusReporter

# clawde's reply palette — named so colours are never magic values (CLAUDE.md).
INLINE_CODE_STYLE = "#c3a6ff"  # pastel purple
CODE_BLOCK_STYLE = "#b5f0a5"  # pastel green
BOLD_STYLE = "bold bright_white"

# Each step carries a leading point — larger than the markdown list marker — that
# is colour-coded: dim while a command runs, then green if it succeeds / red if it
# fails; white while the model writes or thinks.
STEP_GLYPH = "●"
STEP_TEXT_STYLE = "white"
STEP_RUNNING_STYLE = "dim"
STEP_OK_STYLE = "green"
STEP_FAIL_STYLE = "red"
LIST_BULLET = "-"  # markdown unordered-list marker (smaller than the step point)

TOOL_RESULT_GLYPH = "›"
_MAX_ARG_VALUE_LEN = 60
_MAX_RESULT_PREVIEW_LEN = 80
_EMPTY_RESULT = "(no output)"
_MORE = "…"
_ERROR_PREFIX = "Error:"  # how every tool signals failure (tools/base.py)

# The chrome above includes non-ASCII glyphs (●, ›, …); clawde forces UTF-8 output
# at startup (``app._force_utf8``) so it survives a redirected Windows console.

# Per tool: which argument keys to surface, in display order. Presentation only,
# keyed off the tool name — the loop never special-cases a tool (CLAUDE.md). An
# unlisted tool falls back to showing all of its arguments.
_TOOL_ARG_KEYS: dict[str, tuple[str, ...]] = {
    "bash": ("command",),
    "read": ("path", "limit"),
    "write": ("path",),
    "edit": ("path",),
    "glob": ("pattern", "path"),
    "grep": ("pattern", "path"),
}

CLAWDE_THEME = Theme(
    {
        "markdown.code": INLINE_CODE_STYLE,
        "markdown.code_block": CODE_BLOCK_STYLE,
        "markdown.strong": BOLD_STYLE,
        "tool.name": "bold",
        "tool.args": "dim",
        "tool.result": "dim",
    }
)


class _FlatCodeBlock(CodeBlock):
    """A fenced block rendered as flat pastel-green text — no syntax highlighting.

    A deliberate clawde choice: Claude Code leaves blocks uncoloured; we tint the
    whole block one calm colour rather than pygments' rainbow, ignoring any
    ```language hint.
    """

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text(str(self.text).rstrip(), style="markdown.code_block")


class _DashListItem(ListItem):
    """A bullet-list item marked with ``-`` instead of rich's default ``•``.

    Keeps markdown list markers visually distinct from — and smaller than — the
    step point (a larger ``•``).
    """

    def render_bullet(self, console: Console, options: ConsoleOptions) -> RenderResult:
        render_options = options.update(width=options.max_width - 3)
        lines = console.render_lines(self.elements, render_options, style=self.style)
        bullet_style = console.get_style("markdown.item.bullet", default="none")
        bullet = Segment(f" {LIST_BULLET} ", bullet_style)
        padding = Segment(" " * 3, bullet_style)
        new_line = Segment("\n")
        for index, line in enumerate(lines):
            yield bullet if index == 0 else padding
            yield from line
            yield new_line


class ClawdeMarkdown(Markdown):
    """:class:`rich.markdown.Markdown` with flat code blocks and ``-`` list bullets."""

    elements: ClassVar[dict[str, type[MarkdownElement]]] = {
        **Markdown.elements,
        "fence": _FlatCodeBlock,
        "code_block": _FlatCodeBlock,
        "list_item_open": _DashListItem,
    }


def make_console() -> Console:
    """A :class:`rich.console.Console` carrying clawde's reply theme."""
    return Console(theme=CLAWDE_THEME)


def format_tool_call(call: ToolCall) -> Text:
    """Render a call's body as ``name(key="value", …)`` — name bold, args dim."""
    line = Text()
    line.append(call.name, style="tool.name")
    line.append(f"({_argument_summary(call)})", style="tool.args")
    return line


def format_tool_result(result: ToolResult) -> Text:
    """Render a one-line, truncated preview of a result beneath its call."""
    line = Text()
    line.append(f"  {TOOL_RESULT_GLYPH} ", style="tool.result")
    line.append(_result_preview(result.content), style="tool.result")
    return line


def step_block(dot_style: str, body: RenderableType) -> Table:
    """Lay out a step: a coloured leading point in a gutter beside ``body``."""
    grid = Table.grid(padding=(0, 1, 0, 0))
    grid.add_column()
    grid.add_column()
    grid.add_row(Text(STEP_GLYPH, style=dot_style), body)
    return grid


def result_point_style(result: ToolResult) -> str:
    """Green point for a successful tool result, red when it reports an error."""
    return STEP_FAIL_STYLE if result.content.startswith(_ERROR_PREFIX) else STEP_OK_STYLE


def _argument_summary(call: ToolCall) -> str:
    keys = _TOOL_ARG_KEYS.get(call.name)
    if keys is None:
        pairs = list(call.arguments.items())
    else:
        pairs = [(key, call.arguments[key]) for key in keys if key in call.arguments]
    return ", ".join(f"{key}={_short_value(value)}" for key, value in pairs)


def _short_value(value: object) -> str:
    text = value if isinstance(value, str) else str(value)
    truncated = _truncate(text, _MAX_ARG_VALUE_LEN)
    return f'"{truncated}"' if isinstance(value, str) else truncated


def _result_preview(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return _EMPTY_RESULT
    first_line, _, rest = stripped.partition("\n")
    preview = _truncate(first_line, _MAX_RESULT_PREVIEW_LEN)
    if rest and preview == first_line:  # un-truncated, but more lines follow
        preview += f" {_MORE}"
    return preview


def _truncate(text: str, limit: int) -> str:
    collapsed = text.replace("\n", " ")
    if len(collapsed) > limit:
        return collapsed[: limit - 1] + _MORE
    return collapsed


class ReplyRenderer:
    """Streams one turn's reply as a sequence of gutter-pointed, spaced steps.

    A text step streams as live markdown behind a white point; a tool step is
    held until its result is known, then shown behind a green (success) or red
    (failure) point. Between steps the working spinner fills the live region —
    one :class:`Live` at a time, transient so it leaves no trace (#7/#9).
    """

    def __init__(self, console: Console, *, status: StatusReporter | None = None) -> None:
        self._console = console
        self._status = status if status is not None else StatusReporter()
        self._buffer = ""
        self._live: Live | None = None
        self._pending_call: ToolCall | None = None
        self._emitted = False

    def begin(self) -> None:
        """Open the turn: show the working spinner until the first output arrives."""
        self._status.start()
        self._ensure_live()

    def on_text(self, delta: str) -> None:
        """Append a streamed text delta; the spinner gives way to live markdown."""
        if not self._buffer:
            self._separate()
        self._buffer += delta
        self._ensure_live().update(self._renderable(), refresh=True)

    def on_usage(self, usage: Usage) -> None:
        """Update the running token usage shown on the spinner."""
        self._status.update(usage)
        if self._live is not None and not self._buffer:
            self._live.refresh()

    def on_tool_call(self, call: ToolCall) -> None:
        """Show the call launching — live, with the spinner beneath it."""
        self._commit()
        self._separate()
        self._pending_call = call
        self._ensure_live()

    def on_tool_result(self, result: ToolResult) -> None:
        """Replace the launched call with the finished step, pointed green/red."""
        self._commit()  # clear the live "running" call + spinner
        body: RenderableType = format_tool_result(result)
        if self._pending_call is not None:
            body = Group(format_tool_call(self._pending_call), body)
        self._console.print(step_block(result_point_style(result), body))
        self._pending_call = None
        self._emitted = True
        self._ensure_live()

    def finish(self) -> None:
        """Close the turn: commit the final segment and clear any spinner."""
        self._commit()

    def elapsed(self) -> float:
        """Seconds elapsed since the turn began (for the closing summary)."""
        return self._status.elapsed()

    def _ensure_live(self) -> Live:
        if self._live is None:
            if not self._buffer:
                self._status.next_verb()
            self._live = Live(
                self._renderable(),
                console=self._console,
                auto_refresh=True,
                refresh_per_second=REFRESH_PER_SECOND,
                vertical_overflow="visible",
            )
            self._live.start()
        return self._live

    def _renderable(self) -> RenderableType:
        if self._buffer:
            return step_block(STEP_TEXT_STYLE, ClawdeMarkdown(self._buffer))
        if self._pending_call is not None:
            running = step_block(STEP_RUNNING_STYLE, format_tool_call(self._pending_call))
            return Group(running, self._status)
        return self._status

    def _separate(self) -> None:
        # One blank line between steps — but never above the very first one.
        if self._emitted:
            self._console.print()

    def _commit(self) -> None:
        if self._live is None:
            return
        # The live region is always transient; persist a finished text segment by
        # printing it afresh, so it lands in scrollback with a trailing newline on
        # every console (Live.stop() omits that newline on a non-terminal).
        self._live.transient = True
        self._live.stop()
        self._live = None
        if self._buffer:
            self._console.print(self._renderable())
            self._emitted = True
        self._buffer = ""

from __future__ import annotations

from clawde_cli.header import HEADER_STYLE, HeaderState, header_text, render_header


def test_header_text_shows_location_title_model_and_effort() -> None:
    state = HeaderState(
        branch="feature/repl",
        worktree="clawde-wt",
        title="add a REPL",
        provider="anthropic",
        model="claude-opus-4-8",
        effort="high",
    )

    text = header_text(state)

    assert "feature/repl" in text
    assert "clawde-wt" in text
    assert "add a REPL" in text
    assert "anthropic/claude-opus-4-8" in text
    assert "effort high" in text


def test_header_text_falls_back_when_there_is_no_title() -> None:
    state = HeaderState(
        branch="main",
        worktree=None,
        title=None,
        provider="gemini",
        model="gemini-3-pro",
        effort="off",
    )

    text = header_text(state)

    assert "(new session)" in text
    assert "⎇ main" in text  # branch shown without a worktree suffix
    assert "@" not in text  # no worktree -> no "@ folder" suffix


def test_header_text_marks_a_detached_head() -> None:
    state = HeaderState(
        branch=None,
        worktree="clawde-wt",
        title=None,
        provider="openai",
        model="gpt-5",
        effort="off",
    )

    assert "detached @ clawde-wt" in header_text(state)


def test_header_text_handles_no_git_repo() -> None:
    state = HeaderState(
        branch=None,
        worktree=None,
        title=None,
        provider="openai",
        model="gpt-5",
        effort="off",
    )

    assert "no git" in header_text(state)


def test_render_header_carries_the_header_style() -> None:
    state = HeaderState(
        branch="main",
        worktree=None,
        title="t",
        provider="p",
        model="m",
        effort="off",
    )

    rendered = render_header(state)

    assert "main" in rendered.plain
    assert rendered.style == HEADER_STYLE

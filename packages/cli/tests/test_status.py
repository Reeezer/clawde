from __future__ import annotations

from collections.abc import Callable, Sequence

from clawde_core.models import TokenBudget, Usage

from clawde_cli.status import (
    DEFAULT_VERBS,
    SPINNER_FRAME_SECONDS,
    SPINNER_FRAMES,
    StatusReporter,
    format_context,
    format_elapsed,
    format_summary,
    format_token_count,
    spinner_frame,
)


def _clock_returning(*values: float) -> Callable[[], float]:
    ticks = iter(values)

    def clock() -> float:
        return next(ticks)

    return clock


def test_format_elapsed_under_a_minute() -> None:
    assert format_elapsed(36) == "36s"


def test_format_elapsed_minutes_and_seconds() -> None:
    assert format_elapsed(516) == "8m 36s"


def test_format_elapsed_truncates_fractional_seconds() -> None:
    assert format_elapsed(9.9) == "9s"


def test_spinner_frame_starts_at_the_first_frame() -> None:
    assert spinner_frame(0.0) == SPINNER_FRAMES[0]


def test_spinner_frame_cycles_through_every_frame_then_wraps() -> None:
    # A pure function of elapsed time: sampling the middle of each frame's window
    # walks the whole pulse in order, then wraps back round to the first frame.
    half = SPINNER_FRAME_SECONDS / 2
    seen = [spinner_frame(i * SPINNER_FRAME_SECONDS + half) for i in range(len(SPINNER_FRAMES))]
    assert seen == list(SPINNER_FRAMES)
    assert spinner_frame(len(SPINNER_FRAMES) * SPINNER_FRAME_SECONDS + half) == SPINNER_FRAMES[0]


def test_format_token_count_below_one_thousand_is_exact() -> None:
    assert format_token_count(512) == "512"


def test_format_token_count_thousands_use_one_decimal_k() -> None:
    assert format_token_count(46300) == "46.3k"
    assert format_token_count(1000) == "1.0k"


def test_format_token_count_millions_use_one_decimal_m() -> None:
    assert format_token_count(1_048_576) == "1.0M"
    assert format_token_count(2_000_000) == "2.0M"


def test_format_context_reads_used_over_a_rounded_window() -> None:
    # ``used`` keeps its live precision; the window is shown as a round figure.
    assert format_context(TokenBudget(limit=1_048_576, used=24100)) == "ctx 24.1k/1M"
    assert format_context(TokenBudget(limit=200_000, used=512)) == "ctx 512/200k"


def test_format_context_shows_a_placeholder_before_the_first_reply() -> None:
    # Window known upfront, usage not yet measured: a ``–`` stands in for ``used``.
    assert format_context(None, 1_048_576) == "ctx –/1M"
    assert format_context(None, None) is None  # nothing known yet: no ctx at all


def test_format_summary_reads_done_with_time_and_sent_received_tokens() -> None:
    summary = format_summary(516, Usage(input_tokens=12000, output_tokens=46300))
    assert summary.plain == "Done in 8m 36s · ↑ 12.0k ↓ 46.3k tokens"


def test_format_summary_includes_context_budget_when_given() -> None:
    summary = format_summary(
        12,
        Usage(input_tokens=52000, output_tokens=1200),
        TokenBudget(limit=1_048_576, used=35000),
    )
    assert summary.plain == "Done in 12s · ctx 35.0k/1M · ↑ 52.0k ↓ 1.2k tokens"


def test_summary_survives_a_windows_codepage() -> None:
    # The persisted closing line carries clawde's ↑/↓ chrome; clawde forces UTF-8
    # output (app._force_utf8, errors="replace"), so encoding it for a legacy
    # Windows codepage degrades gracefully rather than raising.
    format_summary(5, Usage(input_tokens=5, output_tokens=5)).plain.encode(
        "cp1252", errors="replace"
    )


def test_elapsed_is_zero_before_start_then_counts_up() -> None:
    status = StatusReporter(clock=_clock_returning(10.0, 25.0))
    assert status.elapsed() == 0.0  # before start(), no clock tick consumed
    status.start()
    assert status.elapsed() == 15.0


def test_status_line_reads_like_the_design() -> None:
    status = StatusReporter(
        clock=_clock_returning(100.0, 100.0 + 516),  # start(), then __rich__()
        choose=lambda verbs: "Wandering",
    )
    status.start()
    status.next_verb()
    status.update(Usage(input_tokens=12000, output_tokens=46300))

    assert status.__rich__().plain == (
        f"{spinner_frame(516)} Wandering… (8m 36s · ↑ 12.0k ↓ 46.3k tokens)"
    )


def test_status_line_glyph_pulses_over_time() -> None:
    # Two renders at different elapsed times show different pulse frames; the
    # glyph is the line's first character (a space follows it).
    status = StatusReporter(
        clock=_clock_returning(0.0, 0.0, 1.75),  # start(), then two __rich__() reads
        choose=lambda verbs: "Wandering",
    )
    status.start()
    first = status.__rich__().plain
    later = status.__rich__().plain

    assert first[0] == SPINNER_FRAMES[0]  # elapsed 0.0 → first frame (✢)
    assert later[0] == SPINNER_FRAMES[2]  # elapsed 1.75 → the large frame (✽)
    assert first[0] != later[0]


def test_status_line_shows_pending_context_before_the_first_reply() -> None:
    status = StatusReporter(
        clock=_clock_returning(100.0, 100.0 + 1),  # start(), then __rich__()
        choose=lambda verbs: "Pondering",
    )
    status.start()
    status.next_verb()
    status.set_context_window(1_048_576)  # window seeded; no reply measured yet

    assert status.__rich__().plain == (
        f"{spinner_frame(1)} Pondering… (1s · ctx –/1M · ↑ 0 ↓ 0 tokens)"
    )


def test_status_line_includes_context_when_a_budget_is_set() -> None:
    status = StatusReporter(
        clock=_clock_returning(100.0, 100.0 + 7),  # start(), then __rich__()
        choose=lambda verbs: "Conjuring",
    )
    status.start()
    status.next_verb()
    status.update(Usage(input_tokens=29600, output_tokens=31))
    status.set_budget(TokenBudget(limit=1_048_576, used=24100))

    assert status.__rich__().plain == (
        f"{spinner_frame(7)} Conjuring… (7s · ctx 24.1k/1M · ↑ 29.6k ↓ 31 tokens)"
    )


def test_status_line_before_start_shows_zero_elapsed() -> None:
    status = StatusReporter(clock=_clock_returning(5.0), choose=lambda verbs: "Pondering")
    # No start() — elapsed must read 0s rather than raising.
    assert "(0s · ↑ 0 ↓ 0 tokens)" in status.__rich__().plain


def test_next_verb_draws_from_the_injected_chooser() -> None:
    chosen: list[Sequence[str]] = []

    def choose(verbs: Sequence[str]) -> str:
        chosen.append(verbs)
        return verbs[2]

    status = StatusReporter(verbs=("a", "b", "c"), choose=choose)
    status.next_verb()

    assert chosen == [("a", "b", "c")]
    assert "b…" not in status.__rich__().plain
    assert "c…" in status.__rich__().plain


def test_defaults_use_a_real_clock_and_verb_pool() -> None:
    # Exercises the production defaults (time.monotonic + random.choice) without
    # asserting an exact verb or elapsed value — only that they are well-formed.
    status = StatusReporter()
    status.start()
    status.next_verb()

    rendered = status.__rich__().plain
    assert any(f"{verb}…" in rendered for verb in DEFAULT_VERBS)
    assert "tokens)" in rendered

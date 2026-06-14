from __future__ import annotations

from collections.abc import Callable, Sequence

from clawde_core.models import Usage

from clawde_cli.status import (
    DEFAULT_VERBS,
    SPINNER_GLYPH,
    StatusReporter,
    format_elapsed,
    format_summary,
    format_token_count,
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


def test_format_token_count_below_one_thousand_is_exact() -> None:
    assert format_token_count(512) == "512"


def test_format_token_count_thousands_use_one_decimal_k() -> None:
    assert format_token_count(46300) == "46.3k"
    assert format_token_count(1000) == "1.0k"


def test_format_summary_reads_done_with_time_and_tokens() -> None:
    assert format_summary(516, 46300).plain == "Done in 8m 36s · 46.3k tokens"


def test_summary_middle_dot_survives_a_windows_codepage() -> None:
    format_summary(5, 5).plain.encode("cp1252")  # the closing line is persisted


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
    status.update(Usage(output_tokens=46300))

    assert status.__rich__().plain == f"{SPINNER_GLYPH} Wandering… (8m 36s · ↓ 46.3k tokens)"


def test_status_line_before_start_shows_zero_elapsed() -> None:
    status = StatusReporter(clock=_clock_returning(5.0), choose=lambda verbs: "Pondering")
    # No start() — elapsed must read 0s rather than raising.
    assert "(0s · ↓ 0 tokens)" in status.__rich__().plain


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

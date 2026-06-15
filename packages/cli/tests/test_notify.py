from __future__ import annotations

import io

import pytest
from clawde_core.models import Usage
from rich.console import Console

from clawde_cli.notify import bell_enabled, signal_done
from clawde_cli.session import TurnOutcome, TurnStatus


def _outcome(status: TurnStatus) -> TurnOutcome:
    return TurnOutcome(status=status, usage=Usage())


def _terminal_console() -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    return Console(file=stream, force_terminal=True, width=80), stream


def test_signal_rings_the_bell_when_enabled() -> None:
    console, stream = _terminal_console()

    signal_done(console, _outcome(TurnStatus.OK), bell=True)

    assert "\a" in stream.getvalue()


def test_signal_is_silent_when_the_bell_is_disabled() -> None:
    console, stream = _terminal_console()

    signal_done(console, _outcome(TurnStatus.OK), bell=False)

    assert "\a" not in stream.getvalue()


def test_marker_text_reflects_each_outcome_status() -> None:
    for status, needle in [
        (TurnStatus.OK, "ready"),
        (TurnStatus.ERROR, "failed"),
        (TurnStatus.INTERRUPTED, "stopped"),
    ]:
        console = Console(file=io.StringIO(), record=True, width=80)

        signal_done(console, _outcome(status), bell=False)

        assert needle in console.export_text()


def test_bell_is_enabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAWDE_NO_BELL", raising=False)

    assert bell_enabled() is True


def test_bell_is_disabled_by_the_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAWDE_NO_BELL", "1")

    assert bell_enabled() is False

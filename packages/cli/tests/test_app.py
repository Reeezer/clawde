from __future__ import annotations

from pathlib import Path

import pytest
from clawde_core.models import ImageContent, ReasoningEffort, Usage
from clawde_core.providers.base import ProviderError
from clawde_core.registry import RegistryError
from typer.testing import CliRunner

from clawde_cli import __version__
from clawde_cli import app as app_module
from clawde_cli.app import app
from clawde_cli.session import TurnOutcome, TurnStatus

runner = CliRunner()


class _FakeSession:
    """Stand-in for Session: records the turns it ran and returns a scripted outcome."""

    def __init__(self, outcome: TurnOutcome) -> None:
        self._outcome = outcome
        self.turns: list[tuple[str, tuple[ImageContent, ...]]] = []

    def run_turn(self, prompt: str, images: tuple[ImageContent, ...] = ()) -> TurnOutcome:
        self.turns.append((prompt, tuple(images)))
        return self._outcome


def _ok_outcome() -> TurnOutcome:
    return TurnOutcome(status=TurnStatus.OK, usage=Usage(output_tokens=1))


def _install_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: TurnOutcome | None = None,
    error: Exception | None = None,
) -> tuple[list[dict[str, object]], _FakeSession]:
    """Replace build_session with a stub; return (calls-seen, the fake session)."""
    seen: list[dict[str, object]] = []
    fake = _FakeSession(outcome if outcome is not None else _ok_outcome())

    def fake_build_session(
        console: object,
        *,
        provider: str | None = None,
        model: str | None = None,
        effort: ReasoningEffort | None = None,
    ) -> _FakeSession:
        seen.append({"provider": provider, "model": model, "effort": effort})
        if error is not None:
            raise error
        return fake

    monkeypatch.setattr(app_module, "build_session", fake_build_session)
    return seen, fake


# --- flags & no-arg behaviour -------------------------------------------------


def test_force_utf8_reconfigures_supporting_streams() -> None:
    seen: list[dict[str, str]] = []

    class _Stream:
        def reconfigure(self, *, encoding: str, errors: str) -> None:
            seen.append({"encoding": encoding, "errors": errors})

    app_module._force_utf8(_Stream())

    assert seen == [{"encoding": "utf-8", "errors": "replace"}]


def test_force_utf8_ignores_streams_without_reconfigure() -> None:
    app_module._force_utf8(object())  # no reconfigure attr: a quiet no-op


def test_version_flag_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_prompt_starts_the_repl(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _install_session(monkeypatch)
    monkeypatch.setattr(app_module, "_stdin_is_interactive", lambda: True)
    started: list[object] = []
    monkeypatch.setattr(app_module, "interactive_line_reader", lambda: lambda: "")
    monkeypatch.setattr(
        app_module, "run_repl", lambda session, console, read_line: started.append(session)
    )

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert started == [fake]  # the REPL ran against the built session


def test_repl_build_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_session(monkeypatch, error=ProviderError("Anthropic API key is missing."))
    monkeypatch.setattr(app_module, "_stdin_is_interactive", lambda: True)

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "api key" in result.stdout.lower()


def test_repl_requires_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "_stdin_is_interactive", lambda: False)

    result = runner.invoke(app, [])

    assert result.exit_code == 1
    assert "terminal" in result.stdout.lower()


def test_stdin_is_interactive_returns_a_bool() -> None:
    assert isinstance(app_module._stdin_is_interactive(), bool)


# --- one-shot turn ------------------------------------------------------------


def test_runs_a_turn_and_passes_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _, fake = _install_session(monkeypatch)

    result = runner.invoke(app, ["list files"])

    assert result.exit_code == 0
    assert fake.turns == [("list files", ())]


def test_provider_model_and_effort_flags_reach_build_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen, _ = _install_session(monkeypatch)

    result = runner.invoke(
        app, ["--provider", "anthropic", "--model", "claude-x", "--effort", "high", "hi"]
    )

    assert result.exit_code == 0
    assert seen == [{"provider": "anthropic", "model": "claude-x", "effort": ReasoningEffort.HIGH}]


def test_failed_turn_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_session(
        monkeypatch,
        outcome=TurnOutcome(status=TurnStatus.ERROR, usage=Usage(), message="boom"),
    )

    result = runner.invoke(app, ["do it"])

    assert result.exit_code == 1


def test_provider_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_session(
        monkeypatch,
        error=ProviderError(
            "Anthropic API key is missing. Set CLAWDE_PROVIDERS__ANTHROPIC__API_KEY in your .env."
        ),
    )

    result = runner.invoke(app, ["do something"])

    assert result.exit_code == 1
    assert "api key" in result.stdout.lower()


def test_unknown_provider_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_session(
        monkeypatch,
        error=RegistryError("providers: unknown key 'nope'; known: anthropic, gemini, openai"),
    )

    result = runner.invoke(app, ["--provider", "nope", "hi"])

    assert result.exit_code == 1
    assert "unknown key" in result.stdout.lower()


# --- image loading (validated in the CLI before the turn) ---------------------


def test_image_option_attaches_image_to_the_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, fake = _install_session(monkeypatch)
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    # options precede the prompt, as documented: `clawde --image X "prompt"`
    result = runner.invoke(app, ["--image", str(image_path), "what is this?"])

    assert result.exit_code == 0
    assert len(fake.turns) == 1
    _, images = fake.turns[0]
    assert len(images) == 1
    assert images[0].data == b"\x89PNG\r\n\x1a\nfake"
    assert images[0].mime_type.startswith("image/")  # registry-independent assertion


def test_missing_image_file_is_reported() -> None:
    result = runner.invoke(app, ["--image", "does-not-exist.png", "look"])
    assert result.exit_code == 1
    assert "not found" in result.stdout.lower()


def test_non_image_file_is_reported(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image")

    result = runner.invoke(app, ["--image", str(text_file), "look"])

    assert result.exit_code == 1
    assert "image" in result.stdout.lower()


def test_unreadable_image_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    image_path = tmp_path / "x.png"
    image_path.write_bytes(b"data")

    def deny(self: Path) -> bytes:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", deny)

    result = runner.invoke(app, ["--image", str(image_path), "look"])

    assert result.exit_code == 1
    assert "could not read" in result.stdout.lower()


def test_oversized_image_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(app_module, "_MAX_IMAGE_BYTES", 4)
    image_path = tmp_path / "big.png"
    image_path.write_bytes(b"more than four bytes")

    result = runner.invoke(app, ["--image", str(image_path), "look"])

    assert result.exit_code == 1
    assert "too large" in result.stdout.lower()

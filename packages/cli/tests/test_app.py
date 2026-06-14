from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from clawde_core.loop import AgentError, Turn
from clawde_core.models import ImageContent, ToolCall, Usage
from clawde_core.providers.base import ProviderError
from clawde_core.registry import RegistryError
from typer.testing import CliRunner

from clawde_cli import __version__
from clawde_cli import app as app_module
from clawde_cli.app import app

runner = CliRunner()


def _install_provider(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> list[dict[str, str | None]]:
    """Replace the factory with a stub; return a list recording how it was called."""
    seen: list[dict[str, str | None]] = []

    def fake_build_provider(provider: str | None = None, model: str | None = None) -> object:
        seen.append({"provider": provider, "model": model})
        if error is not None:
            raise error
        return object()  # the faked Agent ignores the provider object

    monkeypatch.setattr(app_module, "build_provider", fake_build_provider)
    return seen


def _fake_agent_class(
    turn: Turn | None = None,
    error: Exception | None = None,
    deltas: tuple[str, ...] = (),
    tool_calls: tuple[ToolCall, ...] = (),
    record_images: list[ImageContent] | None = None,
) -> type:
    class _FakeAgent:
        def __init__(
            self,
            provider: object,
            tools: object,
            *,
            system_prompt: str,
            max_iterations: int = 25,
        ) -> None:
            self.system_prompt = system_prompt

        def stream_turn(
            self,
            user_input: str,
            on_text: Callable[[str], None],
            on_tool_call: Callable[[ToolCall], None] | None = None,
            *,
            images: tuple[ImageContent, ...] = (),
        ) -> Turn:
            if record_images is not None:
                record_images.extend(images)
            if error is not None:
                raise error
            for delta in deltas:
                on_text(delta)
            if on_tool_call is not None:
                for call in tool_calls:
                    on_tool_call(call)
            assert turn is not None
            return turn

    return _FakeAgent


def test_version_flag_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_prompt_shows_a_hint() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "clawde" in result.stdout.lower()


def test_provider_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(
        monkeypatch,
        error=ProviderError(
            "Anthropic API key is missing. Set CLAWDE_PROVIDERS__ANTHROPIC__API_KEY in your .env."
        ),
    )
    result = runner.invoke(app, ["do something"])
    assert result.exit_code == 1
    assert "api key" in result.stdout.lower()


def test_unknown_provider_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(
        monkeypatch,
        error=RegistryError("providers: unknown key 'nope'; known: anthropic, gemini, openai"),
    )
    result = runner.invoke(app, ["--provider", "nope", "hi"])
    assert result.exit_code == 1
    assert "unknown key" in result.stdout.lower()


def test_streams_text_and_tool_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(monkeypatch)
    turn = Turn(
        final_text="here are the files",
        messages=(),
        usage=Usage(input_tokens=3, output_tokens=4),
    )
    agent_cls = _fake_agent_class(
        turn=turn,
        deltas=("here ", "are the files"),
        tool_calls=(ToolCall(id="1", name="bash", arguments={"command": "ls"}),),
    )
    monkeypatch.setattr(app_module, "Agent", agent_cls)

    result = runner.invoke(app, ["list files"])

    assert result.exit_code == 0
    assert "here are the files" in result.stdout  # streamed deltas
    assert "bash" in result.stdout  # tool-call trace
    assert "7 tokens used" in result.stdout


def test_provider_and_model_flags_reach_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_provider(monkeypatch)
    turn = Turn(final_text="ok", messages=(), usage=Usage(output_tokens=1))
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(turn=turn, deltas=("ok",)))

    result = runner.invoke(app, ["--provider", "anthropic", "--model", "claude-x", "hi"])

    assert result.exit_code == 0
    assert seen == [{"provider": "anthropic", "model": "claude-x"}]


def test_agent_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(monkeypatch)
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(error=AgentError("boom")))

    result = runner.invoke(app, ["loop forever"])

    assert result.exit_code == 1
    assert "boom" in result.stdout


def test_image_option_attaches_image_to_the_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_provider(monkeypatch)
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    seen: list[ImageContent] = []
    turn = Turn(final_text="a diagram", messages=(), usage=Usage(output_tokens=1))
    monkeypatch.setattr(
        app_module,
        "Agent",
        _fake_agent_class(turn=turn, deltas=("a diagram",), record_images=seen),
    )

    # options precede the prompt, as documented: `clawde --image X "prompt"`
    result = runner.invoke(app, ["--image", str(image_path), "what is this?"])

    assert result.exit_code == 0
    assert len(seen) == 1
    assert seen[0].data == b"\x89PNG\r\n\x1a\nfake"
    assert seen[0].mime_type.startswith("image/")  # registry-independent assertion


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

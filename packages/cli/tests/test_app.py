from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from clawde_core.loop import AgentError, Turn
from clawde_core.models import (
    CompactionEvent,
    ImageContent,
    ReasoningEffort,
    TokenBudget,
    ToolCall,
    ToolResult,
    Usage,
)
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

    class _StubProvider:
        context_window = 1_048_576  # app reads this to seed the spinner's ctx window

    def fake_build_provider(provider: str | None = None, model: str | None = None) -> object:
        seen.append({"provider": provider, "model": model})
        if error is not None:
            raise error
        return _StubProvider()  # the faked Agent ignores it; app reads context_window

    monkeypatch.setattr(app_module, "build_provider", fake_build_provider)
    return seen


def _fake_agent_class(
    turn: Turn | None = None,
    error: Exception | None = None,
    deltas: tuple[str, ...] = (),
    tool_calls: tuple[ToolCall, ...] = (),
    record_images: list[ImageContent] | None = None,
    budget: TokenBudget | None = None,
    compaction: CompactionEvent | None = None,
) -> type:
    class _FakeAgent:
        def __init__(
            self,
            provider: object,
            tools: object,
            *,
            system_prompt: str,
            max_iterations: int = 25,
            compactor: object = None,
            compaction_threshold: float = 1.0,
        ) -> None:
            self.system_prompt = system_prompt

        def stream_turn(
            self,
            user_input: str,
            on_text: Callable[[str], None],
            on_tool_call: Callable[[ToolCall], None] | None = None,
            on_tool_result: Callable[[ToolResult], None] | None = None,
            on_usage: Callable[[Usage], None] | None = None,
            on_budget: Callable[[TokenBudget], None] | None = None,
            on_compaction: Callable[[CompactionEvent], None] | None = None,
            *,
            images: tuple[ImageContent, ...] = (),
        ) -> Turn:
            if record_images is not None:
                record_images.extend(images)
            if error is not None:
                raise error
            if on_compaction is not None and compaction is not None:
                on_compaction(compaction)
            for delta in deltas:
                on_text(delta)
            for call in tool_calls:
                if on_tool_call is not None:
                    on_tool_call(call)
                if on_tool_result is not None:
                    on_tool_result(ToolResult(tool_call_id=call.id, content="ran ok"))
            if on_usage is not None:
                on_usage(Usage(output_tokens=4))
            if on_budget is not None and budget is not None:
                on_budget(budget)
            assert turn is not None
            return turn

    return _FakeAgent


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
    assert "↑ 3 ↓ 4 tokens" in result.stdout  # closing summary (sent / received)


def test_compaction_event_is_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(monkeypatch)
    turn = Turn(final_text="done", messages=(), usage=Usage(input_tokens=1, output_tokens=1))
    agent_cls = _fake_agent_class(
        turn=turn,
        deltas=("done",),
        compaction=CompactionEvent(
            messages_before=40, messages_after=8, tokens_before=120000, tokens_after=9000
        ),
    )
    monkeypatch.setattr(app_module, "Agent", agent_cls)

    result = runner.invoke(app, ["keep going"])

    assert result.exit_code == 0
    assert "Compacted context" in result.stdout
    assert "40→8 msgs" in result.stdout
    assert "120.0k→9.0k tokens" in result.stdout


def test_summary_shows_the_context_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_provider(monkeypatch)
    turn = Turn(
        final_text="ok",
        messages=(),
        usage=Usage(input_tokens=52000, output_tokens=1200),
    )
    agent_cls = _fake_agent_class(
        turn=turn,
        deltas=("ok",),
        budget=TokenBudget(limit=1_048_576, used=35000),
    )
    monkeypatch.setattr(app_module, "Agent", agent_cls)

    result = runner.invoke(app, ["list files"])

    assert result.exit_code == 0
    assert "ctx 35.0k/1M · ↑ 52.0k ↓ 1.2k tokens" in result.stdout  # ctx joins the summary


def test_provider_and_model_flags_reach_the_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install_provider(monkeypatch)
    turn = Turn(final_text="ok", messages=(), usage=Usage(output_tokens=1))
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(turn=turn, deltas=("ok",)))

    result = runner.invoke(app, ["--provider", "anthropic", "--model", "claude-x", "hi"])

    assert result.exit_code == 0
    assert seen == [{"provider": "anthropic", "model": "claude-x"}]


def test_effort_flag_sets_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[ReasoningEffort] = []

    class _Recorder:
        context_window = 1_048_576  # app reads this to seed the spinner's ctx window

        def set_reasoning_effort(self, effort: ReasoningEffort) -> None:
            recorded.append(effort)

    monkeypatch.setattr(app_module, "build_provider", lambda provider=None, model=None: _Recorder())
    turn = Turn(final_text="ok", messages=(), usage=Usage(output_tokens=1))
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(turn=turn, deltas=("ok",)))

    result = runner.invoke(app, ["--effort", "high", "hi"])

    assert result.exit_code == 0
    assert recorded == [ReasoningEffort.HIGH]


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

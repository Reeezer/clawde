from __future__ import annotations

from collections.abc import Callable

import pytest
from clawde_core.config import GeminiSettings, ProvidersSettings, Settings
from clawde_core.loop import AgentError, Turn
from clawde_core.models import ToolCall, Usage
from typer.testing import CliRunner

from clawde_cli import __version__
from clawde_cli import app as app_module
from clawde_cli.app import app

runner = CliRunner()


def _settings(api_key: str | None, default_model: str | None = None) -> Settings:
    return Settings(
        default_model=default_model,
        providers=ProvidersSettings(gemini=GeminiSettings(api_key=api_key)),
    )


def _fake_agent_class(
    turn: Turn | None = None,
    error: Exception | None = None,
    deltas: tuple[str, ...] = (),
    tool_calls: tuple[ToolCall, ...] = (),
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
        ) -> Turn:
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


def test_missing_api_key_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(api_key=None))
    result = runner.invoke(app, ["do something"])
    assert result.exit_code == 1
    assert "api key" in result.stdout.lower()


def test_streams_text_and_tool_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(api_key="key123"))
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


def test_agent_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(api_key="key123"))
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(error=AgentError("boom")))

    result = runner.invoke(app, ["loop forever"])

    assert result.exit_code == 1
    assert "boom" in result.stdout

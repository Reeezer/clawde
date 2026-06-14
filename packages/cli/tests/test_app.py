from __future__ import annotations

import pytest
from clawde_core.config import GeminiSettings, ProvidersSettings, Settings
from clawde_core.loop import AgentError, Turn
from clawde_core.models import Message, ToolCall, Usage
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


def _fake_agent_class(turn: Turn | None = None, error: Exception | None = None) -> type:
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

        def run_turn(self, user_input: str) -> Turn:
            if error is not None:
                raise error
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


def test_runs_a_turn_and_renders_tool_trace_and_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(api_key="key123"))
    turn = Turn(
        final_text="here are the files",
        messages=(
            Message.assistant(
                content="",
                tool_calls=(ToolCall(id="1", name="bash", arguments={"command": "ls"}),),
            ),
            Message.tool(tool_call_id="1", content="a.py", name="bash"),
            Message.assistant(content="here are the files"),
        ),
        usage=Usage(input_tokens=3, output_tokens=4),
    )
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(turn=turn))

    result = runner.invoke(app, ["list files"])

    assert result.exit_code == 0
    assert "here are the files" in result.stdout
    assert "bash" in result.stdout  # the tool-call trace was rendered


def test_agent_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: _settings(api_key="key123"))
    monkeypatch.setattr(app_module, "Agent", _fake_agent_class(error=AgentError("boom")))

    result = runner.invoke(app, ["loop forever"])

    assert result.exit_code == 1
    assert "boom" in result.stdout

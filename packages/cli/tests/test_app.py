from __future__ import annotations

from typer.testing import CliRunner

from clawde_cli import __version__
from clawde_cli.app import app

runner = CliRunner()


def test_version_command_prints_the_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_a_welcome_hint() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert "clawde" in result.stdout.lower()


def test_help_lists_the_version_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "version" in result.stdout

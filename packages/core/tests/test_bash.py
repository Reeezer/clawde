from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping

import pytest

from clawde_core.tools.bash import BashTool


def _completed(
    stdout: str = "", stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args="cmd", returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_spec_advertises_command_parameter() -> None:
    spec = BashTool().spec
    assert spec.name == "bash"
    assert spec.parameters["required"] == ["command"]


@pytest.mark.parametrize("arguments", [{}, {"command": ""}, {"command": 123}])
def test_run_rejects_invalid_command(
    arguments: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def _fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        return _completed()

    monkeypatch.setattr(subprocess, "run", _fail)
    result = BashTool().run(arguments)
    assert result.startswith("Error:")
    assert called is False


def test_run_prefers_bash_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/bash")
    recorded: dict[str, object] = {}

    def _run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["cmd"] = cmd
        recorded["shell"] = kwargs.get("shell", False)
        return _completed(stdout="hello\n")

    monkeypatch.setattr(subprocess, "run", _run)
    result = BashTool().run({"command": "echo hello"})
    assert result == "hello"
    assert recorded["cmd"] == ["/usr/bin/bash", "-c", "echo hello"]
    assert recorded["shell"] is False


def test_run_falls_back_to_platform_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    recorded: dict[str, object] = {}

    def _run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        recorded["cmd"] = cmd
        recorded["shell"] = kwargs.get("shell", False)
        return _completed()

    monkeypatch.setattr(subprocess, "run", _run)
    result = BashTool().run({"command": "echo hi"})
    assert result == "(no output)"
    assert recorded["cmd"] == "echo hi"
    assert recorded["shell"] is True


def test_run_reports_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/bash")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(stderr="boom", returncode=2))
    result = BashTool().run({"command": "false"})
    assert result == "[exit 2] boom"


def test_run_handles_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: "/bin/bash")

    def _raise(*a: object, **k: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=120)

    monkeypatch.setattr(subprocess, "run", _raise)
    result = BashTool().run({"command": "sleep 999"})
    assert "timed out" in result

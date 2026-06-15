from __future__ import annotations

import subprocess

import pytest

from clawde_cli import gitinfo
from clawde_cli.gitinfo import GitContext, current_git_context


def test_reads_branch_and_worktree_from_git() -> None:
    def fake_git(args: list[str]) -> str | None:
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "feature/repl"
        if args == ["rev-parse", "--show-toplevel"]:
            return "/home/me/code/clawde-wt"
        return None

    context = current_git_context(fake_git)

    assert context == GitContext(branch="feature/repl", worktree="clawde-wt")


def test_missing_repo_degrades_to_none() -> None:
    context = current_git_context(lambda args: None)

    assert context == GitContext(branch=None, worktree=None)


def _completed(
    argv: list[str], *, returncode: int, stdout: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(argv, returncode=returncode, stdout=stdout, stderr="")


def test_run_git_returns_trimmed_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, returncode=0, stdout="main\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gitinfo._run_git(["rev-parse", "--abbrev-ref", "HEAD"]) == "main"


def test_run_git_returns_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, returncode=128, stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gitinfo._run_git(["status"]) is None


def test_run_git_returns_none_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return _completed(argv, returncode=0, stdout="   \n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gitinfo._run_git(["rev-parse", "HEAD"]) is None


def test_run_git_returns_none_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert gitinfo._run_git(["version"]) is None

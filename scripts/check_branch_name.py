#!/usr/bin/env python
"""Validate the current git branch name against clawde's Gitflow convention.

Run by the pre-commit ``pre-push`` hook. See CLAUDE.md "Commit & branch
conventions".
"""

from __future__ import annotations

import re
import subprocess

# Long-lived branches plus the Gitflow working prefixes.
BRANCH = re.compile(r"^(main|develop|(feature|release|hotfix|nightly)/[a-z0-9._/-]+)$")


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> int:
    branch = current_branch()
    if branch in ("HEAD", ""):  # detached HEAD / unborn branch — nothing to check
        return 0
    if BRANCH.match(branch):
        return 0
    print(f"pre-push: branch '{branch}' does not follow the Gitflow convention.\n")
    print("  allowed:  main, develop,")
    print("            feature/<slug>, release/<version>, hotfix/<slug>, nightly/<slug>")
    print(f"  rename:   git branch -m feature/{branch}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

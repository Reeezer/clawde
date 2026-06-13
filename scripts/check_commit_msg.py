#!/usr/bin/env python
"""Validate a commit message against clawde's Conventional Commits convention.

Run by the pre-commit ``commit-msg`` hook, which passes the path to the commit
message as ``argv[1]``. Merge / revert / fixup! / squash! messages pass through
unchanged. See CLAUDE.md "Commit & branch conventions".
"""

from __future__ import annotations

import re
import sys

TYPES = (
    "feat",
    "fix",
    "refactor",
    "docs",
    "test",
    "chore",
    "perf",
    "ci",
    "build",
    "style",
    "revert",
)

# <type>(optional scope)optional-!: description
CONVENTIONAL = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9._/-]+\))?!?: .+")
# Auto-generated messages we never block.
EXEMPT = re.compile(r'^(Merge |Revert "|fixup! |squash! )')

MAX_SUBJECT = 72


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    try:
        with open(path, encoding="utf-8") as handle:
            message = handle.read()
    except OSError as exc:
        print(f"commit-msg: cannot read message file {path!r}: {exc}")
        return 1

    lines = message.strip().splitlines()
    subject = lines[0] if lines else ""

    if EXEMPT.match(subject):
        return 0
    if not CONVENTIONAL.match(subject):
        _explain(subject)
        return 1
    if len(subject) > MAX_SUBJECT:
        print(f"commit-msg: subject is {len(subject)} chars (max {MAX_SUBJECT}):")
        print(f"  {subject}")
        return 1
    return 0


def _explain(subject: str) -> None:
    print("commit-msg: not a Conventional Commit.\n")
    print(f"  got:      {subject!r}\n")
    print("  want:     <type>(<optional scope>): <description>")
    print(f"  types:    {', '.join(TYPES)}")
    print("  examples: feat(core): add ModelProvider ABC")
    print("            fix: handle empty tool result")
    print("  (Merge / Revert / fixup! / squash! commits are exempt.)")


if __name__ == "__main__":
    raise SystemExit(main())

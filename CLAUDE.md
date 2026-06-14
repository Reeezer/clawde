# CLAUDE.md

Guidance for Claude (and other AI agents) working in **clawde**. Keep this file
load-bearing — short, current, specific to *this* repo. It describes what exists
today; forward-looking work lives in [docs/ROADMAP.md](docs/ROADMAP.md).

## What this project is

**clawde** is a from-scratch, **bring-your-own-model** recreation of an agentic
coding CLI (in the spirit of Claude Code), built to *learn the internals the hard
way* and to grow into a complete agentic system. The agent loop is hand-written —
**no agent framework** (LangChain / CrewAI / LlamaIndex) — because understanding
the loop is the point.

It deliberately mirrors the engineering practices of its sibling repo **compass**:
a uv workspace, ruff + mypy `--strict`, a **100% test-coverage gate**, pre-commit,
ADRs, and Gitflow.

- **Roadmap / what's next:** [docs/ROADMAP.md](docs/ROADMAP.md)
- **Domain language:** [CONTEXT.md](CONTEXT.md)
- **Decisions:** [docs/adr/](docs/adr/)

## Architecture (non-negotiable)

The defining pattern, borrowed from compass: **every swappable stage is an ABC
behind a named `Registry`**, resolved by a factory from `Settings`. Adding a model
provider or a tool = subclass the ABC in its own file + register a builder; the
orchestrator never changes. → [ADR-0003](docs/adr/0003-byom-provider-abstraction.md)

1. **ABCs are the contract.** Touch a stage → read its base first.
2. **One implementation per file** (`providers/anthropic.py`, `tools/bash.py`).
3. **Cross-stage data flows through typed models only** — no ad-hoc dicts.
4. **No third-party model SDK at module top.** Import `anthropic` / `openai` /
   `google.genai` *lazily inside* the provider impl, so the base install stays
   light and provider-agnostic and the optional extras stay optional.
5. If you special-case one impl in the loop, the abstraction is wrong — push it
   into the subclass.

## Layout

uv workspace, two packages:

```
packages/core/src/clawde_core/   the engine: registry, config, (soon) models,
                                 providers/, tools/, the agent loop, context
packages/cli/src/clawde_cli/     the terminal app: Typer entry, (soon) REPL,
                                 rendering, slash commands, permission prompts
docs/adr/   docs/ROADMAP.md   CONTEXT.md
```

`clawde-cli` depends on `clawde-core` (workspace source). The engine must stay
usable as a library, independent of the terminal UI.

## Coding style

- **Python 3.12+.** `X | None`, `list[str]`, `StrEnum`, PEP 695 generics
  (`class Registry[T]`, `type Builder[T] = ...`).
- **`from __future__ import annotations`** at the top of every module.
- **Type-annotate everything public.** Pydantic `BaseModel` for cross-stage data,
  `BaseSettings` for config, ABCs for stages.
- **No `typing.Any`** except at untyped-SDK boundaries — mark each with a one-line
  `# untyped: <reason>` and a `[[tool.mypy.overrides]]` entry.
- Narrow custom exceptions. Comments are rare and explain *why*. No dead code.

## Tooling

- **Platform: Windows / PowerShell** (Bash via Git Bash). Python paths via `pathlib`.
- **`uv` for everything** (`uv sync --all-packages`, `uv run …`). No `pip`.
- **`ruff`** (lint + format) and **`mypy --strict`** must be green, per package:
  `uv run --directory packages/<pkg> ruff check . ; … ruff format --check . ; … mypy`.
- **`pytest` with a 100% line-coverage gate** per package (`--cov-fail-under=100`).
- **LF everywhere** (`.gitattributes` + `.editorconfig`).
- **pre-commit:** `uv run pre-commit install` once; runs ruff + the coverage-gated
  pytest suites on commit. mypy is a CI gate.

## Testing

- **Offline by default.** No test hits a real provider: provider impls are tested
  against in-test fakes / recorded responses, and the agent loop against a fake
  provider.
- **Integration tests** are marker-gated (`-m integration`) and skip without a
  provider key — deselected by default, run in CI where keys live.
- **100% coverage is the floor, not the goal** — cover behaviour, not lines.

## Commit & branch conventions

**Branches (Gitflow):** `main` (protected, releases only) ← `release/*` /
`hotfix/*`; `develop` (integration) ← `feature/*`. Names must match `main`,
`develop`, or `(feature|release|hotfix|nightly)/<slug>`.
→ [ADR-0002](docs/adr/0002-gitflow-branching-strategy.md)

**Commits (Conventional Commits):** `<type>(<optional scope>): <description>`,
subject ≤ 72 chars. Types: `feat fix refactor docs test chore perf ci build style
revert`. Merge / `Revert` / `fixup!` / `squash!` commits are exempt.

**Three layers enforce this:**

- `commit-msg` hook → `scripts/check_commit_msg.py` (message format).
- `pre-push` hook → `scripts/check_branch_name.py` (branch name).
- CI `gitflow-guard.yml` (branch name + PR base) **and** GitHub branch protection
  on `main` (PR required, admins included, no force-push/delete) and `develop`
  (CI on PRs, no force-push/delete).

`uv run pre-commit install` registers all three hook types (pre-commit,
commit-msg, pre-push) via `default_install_hook_types`. Re-apply branch
protection with `bash docs/setup-gitflow.sh <owner/repo>`.

## Design discipline

Prioritise design over speed. **Fix the root cause, not the symptom.** Reach for
the right pattern over a fast, smelly patch — clean foundations are cheap to
refactor, bad ones compound. Stay inside the task's boundary, but within it pick
the design that won't need redoing.

## Fact-forcing gate (GateGuard)

A local hook — ECC **GateGuard** — makes you state intent before acting, so edits
stay deliberate. The **first `Bash`** of a session, and the **first `Edit`/`Write`
to each path** (it re-arms occasionally), are blocked once with a request for
facts: who imports the file, the public API affected, any data-file shapes
(synthetic values, never real secrets), and the verbatim user instruction.
**Present the facts, then re-issue the same call** — the retry goes through.

Don't evade it (e.g. writing files through `Bash`/heredoc) — it's a deliberate
guardrail, not an obstacle. The documented escape hatch for bulk setup is
`ECC_GATEGUARD=off` or adding the rule id to `ECC_DISABLED_HOOKS`; ask first.

## What NOT to do

- Don't add an agent framework — the loop is hand-built on purpose.
- Don't import a provider SDK at module top; lazy-import inside the impl.
- Don't let coverage drop below 100%, or commit with red ruff / mypy.
- Don't pass ad-hoc dicts across stage boundaries — add a typed model.
- Don't special-case one provider/tool in the loop — fix the abstraction.
- Don't bypass the commit-msg / branch-name hooks with `--no-verify`.
- Don't evade the fact-forcing gate (GateGuard) — state the facts and retry.

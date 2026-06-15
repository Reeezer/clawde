# clawde

A from-scratch, **bring-your-own-model** coding agent for the terminal — a
learning-grade recreation of an agentic CLI (in the spirit of Claude Code). The
agent loop is hand-written (no agent framework); point it at Anthropic, OpenAI,
Gemini, a local Ollama model, or any OpenAI-compatible endpoint.

> **Status:** Phase 4 — context management (in progress). Phases 0–3 are done: the
> hand-written agent loop, the BYOM providers (Anthropic, OpenAI, Gemini, Ollama, or
> any OpenAI-compatible endpoint) with streaming, multimodal input and reasoning
> effort, and the tool suite (`read`, `write`, `edit`, `bash`, `glob`, `grep`) all
> work today — drive them with `clawde "<prompt>"` and watch a live token/context
> status line. Phase 4 has landed the history store, per-provider token budgeting
> ([ADR-0004](docs/adr/0004-provider-token-counting.md)), and conversation compaction
> ([ADR-0005](docs/adr/0005-conversation-compaction.md)); the interactive REPL is
> Phase 6. See **[docs/ROADMAP.md](docs/ROADMAP.md)**.

## Why

To learn how an agentic coding tool actually works by building one — the loop, the
provider abstraction, the tool system, context management, permissions — and to
grow it into a complete agentic system. See **[CLAUDE.md](CLAUDE.md)** for the
architecture and **[CONTEXT.md](CONTEXT.md)** for the vocabulary.

## Layout

uv workspace, two packages:

- `packages/core` — the engine (`clawde_core`): agent loop, providers (BYOM),
  tools, context.
- `packages/cli` — the terminal app (`clawde_cli`): the `clawde` command.

## Quickstart

Requires [`uv`](https://docs.astral.sh/uv/). On Windows:

```powershell
.\setup.ps1
```

…or manually (any platform):

```bash
uv sync --all-packages
uv run pre-commit install
cp .env.example .env     # then add a provider key (BYOM)
uv run clawde version
```

## Develop

```powershell
# per package: lint, type-check, test (100% coverage gate)
uv run --directory packages/core ruff check . ; uv run --directory packages/core mypy ; uv run --directory packages/core pytest
uv run --directory packages/cli  ruff check . ; uv run --directory packages/cli  mypy ; uv run --directory packages/cli  pytest
```

Gitflow: branch from `develop` as `feature/<name>`; CI and `gitflow-guard` enforce
branch names and PR targets (see
[docs/adr/0002](docs/adr/0002-gitflow-branching-strategy.md)).

### Parallel sessions (git worktrees)

To run several sessions at once — e.g. one Claude Code CLI session per task —
without them trampling each other's files and git state, give each task its own
**git worktree**: a separate working directory backed by the same repo, checked out
to its own branch. Worktrees share one object store, so commits are visible across
them without fetching; integrate as usual by PRing each `feature/*` into `develop`.

Keep every worktree under a single `.worktrees/` directory in the repo root, one
subfolder per branch named after the branch itself (`.worktrees/<branch>`), so
they all live in the same place. `.worktrees/` is gitignored, so these nested
checkouts never show up as untracked in the main clone.

```bash
# from the repo root — one worktree per feature/* branch, all under .worktrees/
git worktree add .worktrees/feature/providers feature/providers
git worktree add .worktrees/feature/repl      feature/repl
```

Each worktree needs its own per-directory setup — `.venv` and `.env` are **not**
shared (both are gitignored):

```bash
cd .worktrees/feature/providers
uv sync --all-packages          # .venv is per-worktree
uv run pre-commit install       # hooks are per-worktree
cp ../../../.env .env           # BYOM key isn't committed — copy it in
```

On Windows, `.\setup.ps1` from inside the worktree does the first two steps and
seeds an empty `.env`; you still copy your keyed `.env` in.

- Two worktrees can't check out the **same** branch — give each its own `feature/*`.
- Keep parallel tasks non-overlapping (e.g. `providers/` vs the CLI) so merges into
  `develop` stay conflict-free.
- Open one terminal + `claude` session per directory; the sessions don't share
  state — only the filesystem and git, which the worktrees keep isolated.
- Done with one? `git worktree remove .worktrees/feature/providers`.

## License

TBD.

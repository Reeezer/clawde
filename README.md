# clawde

A from-scratch, **bring-your-own-model** coding agent for the terminal — a
learning-grade recreation of an agentic CLI (in the spirit of Claude Code). The
agent loop is hand-written (no agent framework); point it at Anthropic, OpenAI,
Gemini, a local Ollama model, or any OpenAI-compatible endpoint.

> **Status:** Phase 0 — foundations. The repo, tooling, and engine/CLI skeletons
> are in place (a `clawde` command and the `Registry[T]` backbone, green under a
> 100% coverage gate). The agent loop and providers are next — see
> **[docs/ROADMAP.md](docs/ROADMAP.md)**.

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

## License

TBD.

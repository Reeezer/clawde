# ADR-0001 — Python 3.12 + uv workspace; ABC / registry / factory architecture

**Status:** Accepted — 2026-06-14

## Context

clawde is a bring-your-own-model coding agent built to learn the internals of an
agentic CLI and grow into a complete system. Two early decisions shape everything
else: the implementation language, and the macro-architecture.

The sibling repo **compass** already encodes a mature, working set of practices
(uv workspace, ruff + mypy strict, 100% coverage, pre-commit, ADRs, Gitflow) and a
clean swappable-stage architecture. Reusing them removes bikeshedding and keeps the
two repos coherent.

**Language.** TypeScript was the obvious alternative — it is what the reference tool
is built in, and the Vercel AI SDK and Ink are excellent. We chose **Python**
because: it fits the owner's existing toolchain (compass); the richest BYOM
ecosystem is in Python; `typer` / `rich` give a strong CLI foundation; and
hand-rolling the provider layer rather than leaning on a framework is the explicit
learning goal — the language should maximise iteration speed on *agent behaviour*,
not systems plumbing.

## Decision

- **Python 3.12+**, managed entirely by **uv**, as a **workspace** with two
  packages: `packages/core` (`clawde-core`, the engine) and `packages/cli`
  (`clawde-cli`, the terminal app). The engine stays usable as a library,
  independent of the UI.
- **Architecture: every swappable stage is an ABC behind a named `Registry`**,
  resolved by a factory from `Settings`. One implementation per file. Cross-stage
  data flows only through typed (pydantic) models. Third-party model SDKs are
  imported lazily inside their impl, behind optional extras. (BYOM specifics in
  ADR-0003.)
- **Quality rails mirror compass:** ruff (lint + format), mypy `--strict`, pytest
  with a 100% line-coverage gate, pre-commit, LF everywhere, ADRs, Gitflow.

## Consequences

- Coherent with compass; moving between the repos needs no retooling.
- The registry/factory pattern makes "add a provider/tool" a local, testable
  change with no orchestrator edits.
- We forgo TypeScript's Ink TUI and the Vercel AI SDK and re-implement provider
  normalisation ourselves — which is the point (learning), at the cost of more code
  than `litellm` would need.
- A future web UI would be a separate package/app (and is where browser E2E /
  Playwright would enter), not a reason to switch the core language.

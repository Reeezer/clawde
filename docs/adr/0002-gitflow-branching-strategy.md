# ADR-0002 — Gitflow branching strategy

**Status:** Accepted — 2026-06-14

## Context

clawde will accumulate parallel work (engine vs CLI, and later sub-agents / MCP /
UI) and possibly unattended agent contributions. It needs an unambiguous rule for
what may land in `main` and how branches are named — the same model already adopted
in compass (its ADR-0011), kept identical so the two repos feel the same.

## Decision

Adopt **Gitflow**.

| Branch | Role | Branches from | Merges to |
|---|---|---|---|
| `main` | Production-ready only; tagged releases | — | — |
| `develop` | Integration; all feature work lands here first | `main` (initial cut) | — |
| `feature/<name>` | New feature or non-urgent fix | `develop` | `develop` via PR |
| `release/<version>` | Release prep (version bump, changelog) | `develop` | `main` via PR, then `develop` backmerge |
| `hotfix/<name>` | Urgent production fix | `main` | `main` via PR, then `develop` backmerge |
| `nightly/<slug>` | Automated overnight PRs (if/when an agent is added) | `develop` | `develop` via PR |

**Enforcement:**

1. **`gitflow-guard.yml`** fails CI when a branch name doesn't match
   `(feature|release|hotfix|nightly)/.+`, or when a PR targets the wrong base
   (`feature/*` & `nightly/*` → `develop`; `hotfix/*` → `main`; `release/*` →
   `main` or `develop`).
2. **Branch protection** on `main` (and PR-gated `develop`), configured once via
   `docs/setup-gitflow.sh` when a GitHub remote exists.

## Consequences

- Every branch carries a Gitflow prefix; invalid names fail CI.
- `main` only ever moves via `release/*` or `hotfix/*`.
- Until a remote + branch protection exist, the guard workflow is dormant, but the
  convention still applies locally — `main` and `develop` are created at init.

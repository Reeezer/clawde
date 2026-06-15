# clawde — Roadmap

A learning-grade, bring-your-own-model coding agent, built one brick at a time.
Ordering is by dependency: each phase needs only the ones above it, so the system
is always runnable. Every phase ships green under the 100% coverage gate.

Legend: ✅ done · 🚧 in progress · 🔜 next · ⏳ later

---

## Phase 0 — Foundations ✅

The repo and its rails. **Done.**

- uv workspace (`packages/core`, `packages/cli`), Python 3.12.
- ruff + mypy `--strict` + pytest 100%-coverage gate, all green.
- pre-commit, CI (cross-platform matrix), Gitflow + guard, ADRs, `.env`/settings.
- `Registry[T]` backbone; a runnable `clawde` command.

## Phase 1 — Minimal agent loop (vertical slice) ✅

The smallest thing that *is* an agent: one hard-coded provider + one tool + the
loop, printing to stdout.

- `models.py` — typed `Message`, `ToolCall`, `ToolResult`, `Usage`.
- A single provider (whichever key you have) and one tool (`bash` / `run`).
- The loop: call → tool calls → execute → feed back → repeat → answer.
- **Done when:** `clawde "list the python files"` completes a real tool-using turn.

## Phase 2 — BYOM provider abstraction ✅

Make the model swappable — the heart of BYOM. → ADR-0003.

- `ModelProvider` ABC + `PROVIDERS` registry + factory from `Settings`.
- Impls (lazy SDK imports, optional extras): Anthropic, Gemini, and one
  OpenAI-compatible provider covering OpenAI + Ollama / local endpoints via a
  configurable `base_url`.
- Normalise messages, **tool-calling (native *and* a JSON-in-text fallback for
  weak/local models)**, streaming, and token usage.
- Multimodal input — images attached to a prompt, normalised per provider. → #6
- Reasoning effort — a normalised `off…max` control mapped to each backend's own
  knob, set per run via `--effort` (the `/effort` REPL toggle lands in Phase 6). → #15
- **Done when:** the same task runs against ≥2 providers via `--provider`.

## Phase 3 — Tool system ✅

- `Tool` ABC + `TOOLS` registry + JSON-Schema params + dispatch / validation. → #2
- Core tools: `read`, `write`, `edit`, `bash`, `glob`, `grep`. → #3 (read/write/glob/grep)
- The `edit` apply logic (exact match → unique-match guard) — the deceptively hard
  one. Shipped exact + unique-guard; whitespace-tolerant *fuzzy* matching is
  deferred to a follow-up (it can silently corrupt files). → #4
- **Done when:** the agent can read, search, and safely edit files end-to-end.

## Phase 4 — Context management 🚧

- History store + per-provider token budgeting — **done.** → ADR-0004, #22
- **Compaction** (summarise old turns; keep recent) — **done.** A `Compactor` ABC +
  `COMPACTORS` registry picks the strategy from settings; the summariser keeps recent
  turns and replaces older ones with a model-written recap when the running estimate
  crosses a threshold (at most once per turn). → ADR-0005, #23
- System-prompt assembly; file-state tracking (don't re-read unchanged files).
- **Done when:** a long session stays under the window without losing the thread.

## Phase 5 — Safety & permissions

- Permission modes (`ask` / `auto` / `plan`), approval prompts.
- Bash sandboxing + allow/deny lists; secret-safe logging.
- Secret-file read protection — tools refuse to read `.env` & configured secrets. → #5
- **Done when:** mutating tools require approval unless explicitly trusted.

## Phase 6 — Interactive REPL / TUI

- A real REPL (prompt_toolkit) with streaming render (rich), markdown, tool-call
  display, slash commands, and clean Ctrl-C interrupts.
- Output presentation, **already shipped** ahead of the full REPL: markdown/colour
  rendering (→ #7 ✅), tool-call formatting (→ #8 ✅), and a live status spinner with
  elapsed time, token totals & context read (→ #9 ✅).
- **Done when:** `clawde` is a pleasant multi-turn interactive session.

## Phase 7 — Persistence, sessions & project memory

- Save / resume sessions; a settings hierarchy; load the repo's `CLAUDE.md`.
- **Done when:** you can quit and resume a conversation, and the agent respects
  project memory.

## Phase 8 — Beyond ⏳

Sub-agents / task delegation · MCP (tool & resource extensibility) · hooks
(lifecycle) · background tasks · plan mode · an optional web UI (where browser /
Playwright E2E would enter) · a small **eval harness** to compare models and
prompts on a fixed task set — the BYOM payoff: measure which model is best for
which job.

---

### Testing posture (every phase)

Offline-first (fake provider, fake tools); integration tests marker-gated behind a
real key; 100% line coverage is the floor. Cover behaviour, not lines.

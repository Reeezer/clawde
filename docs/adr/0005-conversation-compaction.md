# ADR-0005 — Conversation compaction

**Status:** Accepted — 2026-06-15

## Context

Phase 4 keeps a session inside the model's context window. ADR-0004 gave the loop
the two facts it needs — the window size and a running token count — and a
`TokenBudget`. What was missing is the action taken when the budget gets tight:
**compaction**, summarising old turns so a long session stays under the window
without losing the thread.

Several strategies are plausible (summarise via the model; drop the oldest turns;
disable it), and which fits depends on the model and the user — so, like providers
and tools (ADR-0003), compaction is a swappable stage, not a hard-coded routine.

## Decision

- A **`Compactor` ABC** in the `COMPACTORS` registry, resolved from `Settings`
  (`compaction.strategy`) by `build_compactor` — one implementation per file:
  - **`SummarisingCompactor`** (`summarise`, the default): keeps the most recent
    `keep_recent_turns` turns and replaces everything older with a synthetic
    user/assistant **recap** pair, the recap written by one direct
    `provider.complete` call (never through the loop, so compaction can't recurse).
  - **`NoOpCompactor`** (`off`): returns the history unchanged.
- **Cut on turn boundaries.** `History.turns()` groups messages by user-message
  boundary; compaction keeps or drops whole turns, so a tool result is never split
  from the call it answers (which providers reject). The recap is a user→assistant
  pair, so the rewritten history still starts at a user turn and alternates cleanly.
- **Trigger (the loop owns "when").** Before each model call the loop compares its
  running estimate (`TokenBudget.fraction`, from reported usage — free, ADR-0004)
  against `compaction.threshold`. On crossing it asks the compactor once; a real
  compaction spends a single exact `count_tokens` to re-anchor the estimate to the
  smaller conversation.
- **At most once per turn.** A second compaction within the same turn would only
  have the previous recap left to summarise (you cannot summarise the turn you are
  in), so the loop compacts once per turn and detects a strategy's no-op by
  identity — it returns the *same* `History`. Across a session it compacts as often
  as needed: once per turn whenever the estimate is over the threshold.
- **The engine is inert by default.** `Agent` defaults to the no-op compactor at a
  threshold of `1.0`; the CLI wires the settings-configured strategy and threshold.
- A **`CompactionEvent`** carries before/after message and token counts across the
  core→CLI boundary; the CLI renders it as a step so a compaction is visible.

## Consequences

- Adding a compaction strategy is "subclass `Compactor` + register a builder"; the
  loop never changes (ADR-0003 holds).
- Summarisation spends tokens that are **not** folded into the turn's reported
  `Usage` — the budget is a guardrail, not billing (ADR-0004), and the recap is a
  one-off at the threshold.
- A single turn whose own content exceeds the window cannot be compacted (its turn
  is kept intact); that is an inherent limit, not a bug — compaction reduces *prior*
  turns, not the live one.
- `History.since()` is gone: with compaction rewriting history mid-turn, index-based
  slicing is unsound, so the loop tracks each turn's own messages directly.

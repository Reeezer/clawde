# ADR-0004 — Provider token counting & context budgeting

**Status:** Accepted — 2026-06-14

## Context

Phase 4 keeps a session inside the model's context window. To know how close we
are, two facts are needed: the active model's **window size** and the
conversation's **token count**. Both differ per backend — tokenizers and window
limits are vendor- and model-specific — and only the provider knows them
accurately. The loop already receives an exact, free signal after every call:
the provider-reported `input_tokens` / `output_tokens` on each `Completion`.

## Decision

- Extend the **`ModelProvider` ABC** (ADR-0003) with two members:
  - `context_window: int` — the active model's maximum input tokens.
  - `count_tokens(messages, tools) -> int` — the token cost of a conversation.
- **Real providers count exactly:** Anthropic via `messages.count_tokens`,
  Gemini via `models.count_tokens` (both API calls), and the OpenAI-compatible
  provider via **`tiktoken`** (local). The ABC carries a cheap
  ~4-chars-per-token **heuristic fallback** so fake / minimal providers keep the
  contract total and offline tests stay simple.
- **Budget anchoring.** The loop's live, per-step budget is derived from the
  **reported** usage already on each `Completion` (exact, no extra call). The
  exact `count_tokens` is reserved for on-demand / pre-send checks
  (`Agent.budget()`), so the hot loop never makes an extra — possibly online —
  counting request.
- A **`TokenBudget`** value object (`limit` / `used`, with `remaining` and
  `fraction`) carries the signal across the core→CLI boundary; Phase 4
  compaction (#23) is what acts on it.
- **Offline-first tests** (CLAUDE.md). The API-based counters (Anthropic, Gemini)
  are faked in unit tests and exercised for real under `-m integration`;
  `tiktoken` is faked in unit tests to avoid its first-use vocabulary download,
  with a real-encoder test behind the same marker.

## Consequences

- Adding a provider now also means declaring a window and an exact counter — or
  accepting the heuristic fallback inherited from the ABC.
- `tiktoken` joins the `openai` optional extra; the base install stays SDK-free.
- The loop gains `budget()` and an `on_budget` callback; the call contract
  (`complete` / `stream`) is unchanged, so existing providers keep working.
- The budget is a guardrail, not billing: an estimate (or a slightly stale
  reported figure) is acceptable, which is why the hot path avoids online counts.

# ADR-0003 — BYOM provider abstraction

**Status:** Accepted — 2026-06-14

## Context

clawde's headline capability is **bring your own model**: run the same agent
against Anthropic, OpenAI, Gemini, a local Ollama model, or any OpenAI-compatible
endpoint. These backends differ in message shape, tool-calling format, streaming,
token accounting, and reliability — especially tool-calling, where local / weaker
models often can't emit native function calls and must be coaxed into JSON.

## Decision

- A single **`ModelProvider` ABC** defines `complete(messages, tools) -> (text,
  tool_calls, usage)` plus a streaming variant. Concrete providers live
  one-per-file behind the **`PROVIDERS` registry**; a **factory** builds the one
  named in `Settings`.
- **Normalise everything to clawde's own typed models** (`Message`, `ToolCall`,
  `ToolResult`, `Usage`) at the provider boundary — the loop never sees a vendor
  type.
- **Two tool-calling paths, from day one:** the provider's **native**
  function-calling when available, and a **JSON-in-text fallback** (prompt the
  model to emit a tool call as JSON, then parse it) for models without native
  support. A provider declares which it uses.
- **Lazy SDK imports, optional extras.** `anthropic` / `openai` / `google-genai`
  are imported *inside* their impl and shipped as extras; Ollama and
  OpenAI-compatible endpoints use plain HTTP and need no extra. The base install
  and the offline test path stay import-clean.

## Consequences

- Adding a provider = one new file + one `@PROVIDERS.register(...)` + tests; no
  loop changes.
- The fallback path lets weak / local models participate (the real BYOM payoff),
  at the cost of more parsing and robustness work — tracked in Phase 2.
- We own the normalisation surface (more code than adopting `litellm`), which is
  the deliberate learning trade chosen in ADR-0001.
- Secrets come only from `CLAWDE_PROVIDERS__<NAME>__API_KEY` (env / `.env`), never
  hard-coded.

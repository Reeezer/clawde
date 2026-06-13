# CONTEXT.md — clawde's ubiquitous language

The vocabulary every module, doc, and commit should use. (Mirrors compass's
CONTEXT.md in spirit: name the domain once, then use it everywhere.)

## The agent

- **Agent** — the thing that pursues a user's goal by thinking and acting in a
  loop. clawde is one agent (sub-agents come later).
- **Agent loop** — the core cycle: assemble context → call the model → if it
  returns tool calls, execute them and feed the results back → repeat until it
  answers. The heart of the system; hand-written, no framework.
- **Turn** — one user input and everything the agent does in response (one or many
  model calls + tool executions) up to the next time it yields to the user.

## Model & provider (BYOM)

- **Provider** — an adapter to one model backend (Anthropic, OpenAI, Gemini,
  Ollama, any OpenAI-compatible endpoint). Lives behind the `ModelProvider` ABC in
  the `PROVIDERS` registry. *Bring your own model* = choose a provider + key.
- **Model** — a specific model id served by a provider (e.g. `claude-opus-4-8`).
- **Message** — one entry in the conversation: a role (`system` / `user` /
  `assistant` / `tool`) plus content. Providers normalise their wire format
  to/from these.
- **Usage** — token counts (input / output / cached) a provider reports per call.

## Tools

- **Tool** — a capability the model can invoke (read a file, run a command,
  search). Lives behind the `Tool` ABC in the `TOOLS` registry, with a JSON-Schema
  parameter spec.
- **Tool call** — the model's request to run a tool with arguments.
- **Tool result** — the output handed back to the model as the next message.

## Context & memory

- **Context window** — the bounded set of tokens sent to the model each call:
  system prompt + tool specs + conversation history.
- **Compaction** — summarising/dropping old history when it no longer fits the
  window, preserving recent and pinned content.
- **Session** — one persisted conversation (history + state) that can be resumed.
- **Project memory** — repo-local guidance the agent loads (clawde's own
  `CLAUDE.md` convention).

## Control

- **Permission / permission mode** — the gate deciding whether a tool runs
  automatically, needs approval, or is denied (e.g. `ask` / `auto` / `plan`).
- **Registry / factory** — `Registry[T]` maps a name to a builder; the factory
  reads `Settings` and builds the chosen provider/tool. → ADR-0003.
- **Settings** — typed configuration from `CLAWDE_`-prefixed env / `.env`.

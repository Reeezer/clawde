r"""JSON-in-text tool-calling fallback (ADR-0003, issue #21).

Wraps any :class:`~clawde_core.providers.base.ModelProvider` so a model *without*
native function-calling — typically a local or weaker one — can still drive the
agent loop. The wrapper:

* injects the available tools and a JSON convention into the system prompt;
* flattens clawde's typed tool calls and tool results into plain text the model
  can read (it has no native tool role);
* sends the conversation with *no* native tools, then parses fenced ``json``
  tool-call blocks out of the reply back into typed :class:`ToolCall`\ s.

The agent loop is untouched: it sees the same :class:`Completion` whether the
provider called tools natively or through this fallback (CLAUDE.md).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from clawde_core.models import Completion, Message, Role, ToolCall, ToolSpec
from clawde_core.providers.base import ModelProvider

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JsonToolCallingProvider(ModelProvider):
    """Gives an inner provider JSON-in-text tool-calling (for models lacking native).

    Composition, not inheritance: it wraps ``inner`` and reshapes the
    conversation around it, so any provider gains the fallback unchanged.
    Streaming uses the ABC default (one terminal chunk) — the whole reply is
    needed before its tool-call JSON can be parsed.
    """

    def __init__(self, inner: ModelProvider) -> None:
        self._inner = inner

    def complete(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> Completion:
        completion = self._inner.complete(_shape(messages, tools), [])
        if not tools:
            return completion
        text, tool_calls = _parse(completion.text)
        return Completion(text=text, tool_calls=tool_calls, usage=completion.usage)


def _shape(messages: Sequence[Message], tools: Sequence[ToolSpec]) -> list[Message]:
    """Rewrite the conversation for a model with no native tool support."""
    protocol = _protocol(tools) if tools else None
    shaped: list[Message] = []
    injected = False
    for message in messages:
        if message.role is Role.SYSTEM and protocol is not None and not injected:
            body = f"{message.content}\n\n{protocol}" if message.content else protocol
            shaped.append(Message.system(body))
            injected = True
        elif message.role is Role.ASSISTANT:
            shaped.append(_flatten_assistant(message))
        elif message.role is Role.TOOL:
            shaped.append(_flatten_tool(message))
        else:
            shaped.append(message)
    if protocol is not None and not injected:
        shaped.insert(0, Message.system(protocol))
    return shaped


def _protocol(tools: Sequence[ToolSpec]) -> str:
    lines = [
        "You have tools, but you must call them as TEXT (there is no native tool API).",
        "To call a tool, reply with ONLY a fenced json block, nothing else:",
        "```json",
        '{"tool": "<tool_name>", "arguments": {<arguments>}}',
        "```",
        "Call one tool at a time; its output comes back as a 'TOOL RESULT' message. "
        "When you are finished, reply normally with your final answer and no json block.",
        "",
        "Available tools:",
    ]
    for spec in tools:
        lines.append(f"- {spec.name}: {spec.description}")
        lines.append(f"  arguments JSON Schema: {json.dumps(spec.parameters)}")
    return "\n".join(lines)


def _flatten_assistant(message: Message) -> Message:
    """Render a prior assistant turn (its text plus any tool calls) as plain text."""
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    parts.extend(_render_call(call) for call in message.tool_calls)
    return Message.assistant(content="\n".join(parts))


def _render_call(call: ToolCall) -> str:
    payload = {"tool": call.name, "arguments": dict(call.arguments)}
    return f"```json\n{json.dumps(payload)}\n```"


def _flatten_tool(message: Message) -> Message:
    """Render a tool result as a user message (the model has no tool role)."""
    return Message.user(f"TOOL RESULT ({message.name or 'tool'}):\n{message.content}")


def _parse(text: str) -> tuple[str, tuple[ToolCall, ...]]:
    """Extract fenced-json tool calls from ``text``; return ``(clean_text, calls)``."""
    calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    for match in _FENCE.finditer(text):
        call = _as_tool_call(match.group(1), len(calls))
        if call is not None:
            calls.append(call)
            spans.append(match.span())
    if calls:
        return _without(text, spans).strip(), tuple(calls)
    bare = _as_tool_call(text, 0)
    if bare is not None:
        return "", (bare,)
    return text.strip(), ()


def _as_tool_call(snippet: str, index: int) -> ToolCall | None:
    """Parse one JSON snippet into a :class:`ToolCall`, or ``None`` if it isn't one."""
    try:
        payload: object = json.loads(snippet.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("tool")
    if not isinstance(name, str):
        return None
    arguments = payload.get("arguments", {})
    return ToolCall(
        id=f"{name}-{index}",
        name=name,
        arguments=arguments if isinstance(arguments, dict) else {},
    )


def _without(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Return ``text`` with the given ``(start, end)`` character spans removed."""
    kept: list[str] = []
    last = 0
    for start, end in spans:
        kept.append(text[last:start])
        last = end
    kept.append(text[last:])
    return "".join(kept)

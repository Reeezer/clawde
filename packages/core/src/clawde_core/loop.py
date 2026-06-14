"""The agent loop — clawde's beating heart.

One hand-written cycle, no framework: assemble the conversation, call the model,
and while it asks for tools, execute them and feed the results back, until it
answers. The loop depends only on the
:class:`~clawde_core.providers.base.ModelProvider` and
:class:`~clawde_core.tools.base.Tool` ABCs — swapping the model or adding a tool
never touches this file (see ``CLAUDE.md``, ADR-0003).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from clawde_core.models import Message, ToolCall, ToolResult, ToolSpec, Usage
from clawde_core.providers.base import ModelProvider
from clawde_core.tools.base import Tool

DEFAULT_MAX_ITERATIONS = 25


class AgentError(RuntimeError):
    """The loop could not complete the turn within its step budget."""


class Turn(BaseModel):
    """Everything the agent did in response to one user input."""

    model_config = ConfigDict(frozen=True)

    final_text: str
    messages: tuple[Message, ...]
    usage: Usage


class Agent:
    """Runs the agent loop against a provider and a fixed set of tools."""

    def __init__(
        self,
        provider: ModelProvider,
        tools: Sequence[Tool],
        *,
        system_prompt: str,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._provider = provider
        self._tools = {tool.spec.name: tool for tool in tools}
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._history: list[Message] = []

    def run_turn(self, user_input: str) -> Turn:
        """Drive one turn to completion and return what happened."""
        turn_start = len(self._history)
        self._history.append(Message.user(user_input))
        usage = Usage()
        for _ in range(self._max_iterations):
            completion = self._provider.complete(self._conversation(), self._specs())
            usage += completion.usage
            self._history.append(
                Message.assistant(content=completion.text, tool_calls=completion.tool_calls)
            )
            if not completion.tool_calls:
                return Turn(
                    final_text=completion.text,
                    messages=tuple(self._history[turn_start:]),
                    usage=usage,
                )
            for call in completion.tool_calls:
                result = self._execute(call)
                self._history.append(
                    Message.tool(
                        tool_call_id=result.tool_call_id, content=result.content, name=call.name
                    )
                )
        raise AgentError(f"agent did not answer within {self._max_iterations} iterations")

    def _conversation(self) -> list[Message]:
        return [Message.system(self._system_prompt), *self._history]

    def _specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def _execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(
                tool_call_id=call.id,
                content=f"Error: unknown tool '{call.name}'. Available tools: {available}.",
            )
        return ToolResult(tool_call_id=call.id, content=tool.run(call.arguments))

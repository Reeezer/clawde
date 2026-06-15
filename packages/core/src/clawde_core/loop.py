"""The agent loop — clawde's beating heart.

One hand-written cycle, no framework: assemble the conversation, call the model,
and while it asks for tools, execute them and feed the results back, until it
answers. :meth:`Agent.run_turn` collects the whole reply; :meth:`Agent.stream_turn`
surfaces it token-by-token as it arrives. The loop depends only on the
:class:`~clawde_core.providers.base.ModelProvider` and
:class:`~clawde_core.tools.base.Tool` ABCs — swapping the model or adding a tool
never touches this file (see ``CLAUDE.md``, ADR-0003).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from clawde_core.context.history import History
from clawde_core.models import (
    Completion,
    ImageContent,
    Message,
    TokenBudget,
    ToolCall,
    ToolResult,
    ToolSpec,
    Usage,
)
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
        self._history = History()

    def clear(self) -> None:
        """Drop the conversation, starting the next turn fresh (the ``/clear`` command)."""
        self._history = History()

    def run_turn(self, user_input: str, *, images: tuple[ImageContent, ...] = ()) -> Turn:
        """Drive one turn to completion, collecting the whole reply."""
        return self._drive(
            user_input,
            lambda: self._provider.complete(self._conversation(), self._specs()),
            images=images,
        )

    def stream_turn(
        self,
        user_input: str,
        on_text: Callable[[str], None],
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
        on_budget: Callable[[TokenBudget], None] | None = None,
        *,
        images: tuple[ImageContent, ...] = (),
    ) -> Turn:
        """Drive one turn, streaming text deltas to ``on_text`` as they arrive,
        announcing each tool call to ``on_tool_call`` before it runs, each
        :class:`ToolResult` to ``on_tool_result`` once it has, the running
        :class:`Usage` to ``on_usage`` after each model call, and the running
        :class:`TokenBudget` to ``on_budget`` alongside it."""
        return self._drive(
            user_input,
            lambda: self._stream_completion(on_text),
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_usage=on_usage,
            on_budget=on_budget,
            images=images,
        )

    def _drive(
        self,
        user_input: str,
        next_completion: Callable[[], Completion],
        on_tool_call: Callable[[ToolCall], None] | None = None,
        on_tool_result: Callable[[ToolResult], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
        on_budget: Callable[[TokenBudget], None] | None = None,
        *,
        images: tuple[ImageContent, ...] = (),
    ) -> Turn:
        turn_start = len(self._history)
        self._history.append(Message.user(user_input, images=images))
        usage = Usage()
        for _ in range(self._max_iterations):
            completion = next_completion()
            usage += completion.usage
            if on_usage is not None:
                on_usage(usage)
            if on_budget is not None:
                on_budget(self._reported_budget(completion.usage))
            self._history.append(
                Message.assistant(
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                    thinking=completion.thinking,
                )
            )
            if not completion.tool_calls:
                return Turn(
                    final_text=completion.text,
                    messages=self._history.since(turn_start),
                    usage=usage,
                )
            for call in completion.tool_calls:
                if on_tool_call is not None:
                    on_tool_call(call)
                result = self._execute(call)
                if on_tool_result is not None:
                    on_tool_result(result)
                self._history.append(
                    Message.tool(
                        tool_call_id=result.tool_call_id, content=result.content, name=call.name
                    )
                )
        raise AgentError(f"agent did not answer within {self._max_iterations} iterations")

    def _stream_completion(self, on_text: Callable[[str], None]) -> Completion:
        final: Completion | None = None
        for chunk in self._provider.stream(self._conversation(), self._specs()):
            if chunk.text:
                on_text(chunk.text)
            if chunk.completion is not None:
                final = chunk.completion
        if final is None:
            raise AgentError("provider stream ended without a completion")
        return final

    def budget(self) -> TokenBudget:
        """The exact current context budget, counting the live conversation.

        Unlike the per-step ``on_budget`` signal (which rides on the usage each
        completion reports), this asks the provider to count tokens and so may
        call the provider's API (ADR-0004) — invoke it deliberately, not on
        every render.
        """
        return TokenBudget(
            limit=self._provider.context_window,
            used=self._provider.count_tokens(self._conversation(), self._specs()),
        )

    def _reported_budget(self, usage: Usage) -> TokenBudget:
        return TokenBudget(
            limit=self._provider.context_window,
            used=usage.input_tokens + usage.output_tokens,
        )

    def _conversation(self) -> list[Message]:
        return [Message.system(self._system_prompt), *self._history.messages]

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
        return ToolResult(tool_call_id=call.id, content=tool.invoke(call.arguments))

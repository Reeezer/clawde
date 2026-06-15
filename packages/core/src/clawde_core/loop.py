"""The agent loop — clawde's beating heart.

One hand-written cycle, no framework: assemble the conversation, call the model,
and while it asks for tools, execute them and feed the results back, until it
answers. :meth:`Agent.run_turn` collects the whole reply; :meth:`Agent.stream_turn`
surfaces it token-by-token as it arrives. The loop depends only on the
:class:`~clawde_core.providers.base.ModelProvider`,
:class:`~clawde_core.tools.base.Tool`, and
:class:`~clawde_core.context.compaction.base.Compactor` ABCs — swapping the model,
adding a tool, or changing the compaction strategy never touches this file (see
``CLAUDE.md``, ADR-0003). When the running token estimate crosses
``compaction_threshold`` it asks the compactor to shrink the history (ADR-0005).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

from clawde_core.context.compaction.base import Compactor
from clawde_core.context.compaction.off import NoOpCompactor
from clawde_core.context.history import History
from clawde_core.models import (
    CompactionEvent,
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
        compactor: Compactor | None = None,
        compaction_threshold: float = 1.0,
    ) -> None:
        """``compactor`` shrinks the history when the running token estimate reaches
        ``compaction_threshold`` (a fraction of the window). The defaults — a no-op
        compactor at a threshold of ``1.0`` — leave compaction off; the CLI wires the
        settings-configured strategy and threshold (ADR-0005)."""
        self._provider = provider
        self._tools = {tool.spec.name: tool for tool in tools}
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._compactor = compactor if compactor is not None else NoOpCompactor()
        self._compaction_threshold = compaction_threshold
        self._history = History()
        self._budget_estimate = TokenBudget(limit=provider.context_window, used=0)

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
        on_compaction: Callable[[CompactionEvent], None] | None = None,
        *,
        images: tuple[ImageContent, ...] = (),
    ) -> Turn:
        """Drive one turn, streaming text deltas to ``on_text`` as they arrive,
        announcing each tool call to ``on_tool_call`` before it runs, each
        :class:`ToolResult` to ``on_tool_result`` once it has, the running
        :class:`Usage` to ``on_usage`` after each model call, the running
        :class:`TokenBudget` to ``on_budget`` alongside it, and a
        :class:`CompactionEvent` to ``on_compaction`` whenever the history is
        compacted."""
        return self._drive(
            user_input,
            lambda: self._stream_completion(on_text),
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_usage=on_usage,
            on_budget=on_budget,
            on_compaction=on_compaction,
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
        on_compaction: Callable[[CompactionEvent], None] | None = None,
        *,
        images: tuple[ImageContent, ...] = (),
    ) -> Turn:
        turn: list[Message] = []
        self._record(Message.user(user_input, images=images), turn)
        usage = Usage()
        compacted_this_turn = False
        for _ in range(self._max_iterations):
            if not compacted_this_turn and self._over_budget():
                compacted_this_turn = True
                event = self._compact()
                if event is not None and on_compaction is not None:
                    on_compaction(event)
            completion = next_completion()
            usage += completion.usage
            if on_usage is not None:
                on_usage(usage)
            self._budget_estimate = self._reported_budget(completion.usage)
            if on_budget is not None:
                on_budget(self._budget_estimate)
            self._record(
                Message.assistant(
                    content=completion.text,
                    tool_calls=completion.tool_calls,
                    thinking=completion.thinking,
                ),
                turn,
            )
            if not completion.tool_calls:
                return Turn(final_text=completion.text, messages=tuple(turn), usage=usage)
            for call in completion.tool_calls:
                if on_tool_call is not None:
                    on_tool_call(call)
                result = self._execute(call)
                if on_tool_result is not None:
                    on_tool_result(result)
                self._record(
                    Message.tool(
                        tool_call_id=result.tool_call_id, content=result.content, name=call.name
                    ),
                    turn,
                )
        raise AgentError(f"agent did not answer within {self._max_iterations} iterations")

    def _record(self, message: Message, turn: list[Message]) -> None:
        """Append a message to the running history and to this turn's own record.

        The ``turn`` list is the loop's ground truth for what the current turn
        produced, independent of compaction rewriting the history mid-turn.
        """
        self._history.append(message)
        turn.append(message)

    def _over_budget(self) -> bool:
        return self._budget_estimate.fraction >= self._compaction_threshold

    def _compact(self) -> CompactionEvent | None:
        """Ask the compactor to shrink the history; return what it did, or ``None``.

        A strategy that can't shrink the conversation returns it unchanged, which
        the loop reports as ``None`` and does not retry this turn. Only a real
        compaction spends one exact ``count_tokens`` (ADR-0004), to re-anchor the
        running estimate to the smaller conversation.
        """
        before = len(self._history)
        compacted = self._compactor.compact(self._history, provider=self._provider)
        if compacted is self._history:  # the strategy left it unchanged — nothing to do
            return None
        tokens_before = self._budget_estimate.used
        self._history = compacted
        tokens_after = self._provider.count_tokens(self._conversation(), self._specs())
        self._budget_estimate = TokenBudget(limit=self._provider.context_window, used=tokens_after)
        return CompactionEvent(
            messages_before=before,
            messages_after=len(self._history),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

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

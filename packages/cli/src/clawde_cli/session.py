"""A clawde session: the provider, agent, and running totals behind one turn.

Both entry points share this. The one-shot ``clawde "task"`` builds a Session
and runs a single turn; the interactive REPL (#10) builds one and runs many. A
Session drives a turn — streaming the reply through the renderer and printing the
closing summary — and returns a typed :class:`TurnOutcome`; it never exits the
process, so the caller decides what a failed turn means (the one-shot exits
non-zero, the REPL prints the error and waits for the next prompt). The slash
commands (#11) act on the Session too: clear the history, toggle reasoning
effort, read the running usage.
"""

from __future__ import annotations

import contextlib
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from clawde_core.config import get_settings
from clawde_core.context.compaction.factory import build_compactor
from clawde_core.loop import Agent, AgentError
from clawde_core.models import CompactionEvent, ImageContent, ReasoningEffort, TokenBudget, Usage
from clawde_core.providers.base import ModelProvider, ProviderError
from clawde_core.providers.factory import build_provider
from clawde_core.tools.registry import build_tools
from rich.console import Console

from clawde_cli.rendering import ReplyRenderer
from clawde_cli.status import format_summary

_TITLE_MAX_LEN = 50


class TurnStatus(StrEnum):
    """How a turn ended — drives the done signal (#13) and the REPL's next step."""

    OK = "ok"
    ERROR = "error"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class TurnOutcome:
    """The result of one turn: how it ended, its token usage, and any error text."""

    status: TurnStatus
    usage: Usage
    message: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the turn completed normally."""
        return self.status is TurnStatus.OK


def default_system_prompt() -> str:
    """clawde's system prompt — names the platform and the available tools."""
    return (
        "You are clawde, a coding agent working in a terminal on "
        f"{platform.system()}. You have tools to read, write, and edit files, to "
        "find files (glob) and search their contents (grep), and to run shell "
        "commands (bash). Use them to inspect the project and accomplish the "
        "user's request, then reply with a concise final answer."
    )


class Session:
    """One clawde session: a provider + agent and the running totals a REPL shows.

    Drives a turn at a time via :meth:`run_turn`, accumulating token usage and
    remembering the latest context read so ``/usage`` and the status header have
    something to show. Stage objects (the provider, the agent) are injected, so
    the engine stays UI-free and the Session is easy to drive under test.
    """

    def __init__(
        self,
        *,
        console: Console,
        provider: ModelProvider,
        provider_name: str,
        agent: Agent,
    ) -> None:
        self._console = console
        self._provider = provider
        self._provider_name = provider_name
        self._agent = agent
        self._usage = Usage()
        self._budget: TokenBudget | None = None
        self._title: str | None = None

    @property
    def provider_name(self) -> str:
        """The resolved provider key (e.g. ``anthropic``)."""
        return self._provider_name

    @property
    def model(self) -> str:
        """The active model id (e.g. ``claude-opus-4-8``)."""
        return self._provider.model

    @property
    def effort(self) -> ReasoningEffort:
        """The reasoning effort applied to subsequent turns."""
        return self._provider.reasoning_effort

    @property
    def usage(self) -> Usage:
        """Cumulative token usage across every completed turn this session."""
        return self._usage

    @property
    def context_budget(self) -> TokenBudget | None:
        """The most recent context read (``ctx used/limit``), or ``None`` before the first turn."""
        return self._budget

    @property
    def title(self) -> str | None:
        """A short label for the session, derived from the first prompt."""
        return self._title

    def set_effort(self, effort: ReasoningEffort) -> None:
        """Change the reasoning effort for subsequent turns (the ``/effort`` toggle)."""
        self._provider.set_reasoning_effort(effort)

    def clear(self) -> None:
        """Reset the conversation, keeping the session's running totals (``/clear``)."""
        self._agent.clear()

    def compact(self) -> CompactionEvent | None:
        """Compact the conversation now (the ``/compact`` command); returns what
        changed, or ``None`` when there was nothing old enough to compact."""
        return self._agent.compact()

    def run_turn(self, prompt: str, images: Sequence[ImageContent] = ()) -> TurnOutcome:
        """Drive one turn: stream the reply, print the summary, return the outcome.

        Never raises for an expected failure: a provider/agent error or a Ctrl-C
        interrupt is caught, the live region is closed, and the cause is reported
        in the outcome — so a REPL can show it and read the next prompt.
        """
        if self._title is None:
            self._title = _derive_title(prompt)
        renderer = ReplyRenderer(self._console)
        renderer.begin(self._provider.context_window)
        try:
            turn = self._agent.stream_turn(
                prompt,
                on_text=renderer.on_text,
                on_tool_call=renderer.on_tool_call,
                on_tool_result=renderer.on_tool_result,
                on_usage=renderer.on_usage,
                on_budget=renderer.on_budget,
                on_compaction=renderer.on_compaction,
                images=tuple(images),
            )
        except KeyboardInterrupt:
            # Usage is reported as zero on a failed turn: clawde does not count
            # the tokens of an incomplete turn against the session.
            _finish_quietly(renderer)
            self._console.print("\n[dim]Interrupted — the turn was cancelled.[/dim]")
            return TurnOutcome(status=TurnStatus.INTERRUPTED, usage=Usage())
        except (ProviderError, AgentError) as exc:
            _finish_quietly(renderer)
            self._console.print(f"\n[red]Error:[/red] {exc}")
            return TurnOutcome(status=TurnStatus.ERROR, usage=Usage(), message=str(exc))
        renderer.finish()
        self._usage += turn.usage
        self._budget = renderer.context_budget()
        self._console.print()
        self._console.print(
            format_summary(renderer.elapsed(), turn.usage, renderer.context_budget())
        )
        return TurnOutcome(status=TurnStatus.OK, usage=turn.usage)


def build_session(
    console: Console,
    *,
    provider: str | None = None,
    model: str | None = None,
    effort: ReasoningEffort | None = None,
) -> Session:
    """Build a Session from settings, optionally overriding provider/model/effort.

    Raises :class:`~clawde_core.providers.base.ProviderError` or
    :class:`~clawde_core.registry.RegistryError` when the provider can't be
    built; the caller turns that into a friendly message and exits.
    """
    settings = get_settings()
    model_provider = build_provider(provider, model)
    if effort is not None:
        model_provider.set_reasoning_effort(effort)
    agent = Agent(
        model_provider,
        build_tools(settings),
        system_prompt=default_system_prompt(),
        compactor=build_compactor(settings),
        compaction_threshold=settings.compaction.threshold,
    )
    return Session(
        console=console,
        provider=model_provider,
        provider_name=provider or settings.default_provider,
        agent=agent,
    )


def _finish_quietly(renderer: ReplyRenderer) -> None:
    """Close the renderer's live region while handling a failed turn, ignoring
    any rendering error so the turn still returns its outcome rather than masking
    it with a cleanup exception."""
    with contextlib.suppress(Exception):
        renderer.finish()


def _derive_title(prompt: str) -> str:
    """A short, single-line title from the first prompt (empty if it is blank)."""
    stripped = prompt.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0]
    if len(first_line) > _TITLE_MAX_LEN:
        return first_line[: _TITLE_MAX_LEN - 1] + "…"
    return first_line

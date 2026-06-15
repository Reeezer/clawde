"""In-session slash commands (#11, #15).

Typed at the REPL prompt and intercepted before they ever reach the model. A
small registry maps each ``/name`` to a :class:`SlashCommand`, so adding one is
local — write the class, decorate it — and the loop never changes (CLAUDE.md).
Each command acts on the live :class:`~clawde_cli.session.Session` (clear the
history, toggle reasoning effort, read usage) and returns a typed
:class:`CommandResult` telling the REPL what to show and whether to stop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from clawde_core.models import ReasoningEffort
from rich.console import RenderableType
from rich.text import Text

from clawde_cli.rendering import format_compaction
from clawde_cli.session import Session
from clawde_cli.status import RECEIVED_GLYPH, SENT_GLYPH, format_context, format_token_count

_INFO_STYLE = "#f5e0a3"  # pastel yellow, matching the turn summary
_NOTE_STYLE = "dim"
_WARN_STYLE = "yellow"
_ERROR_STYLE = "red"


@dataclass(frozen=True)
class CommandResult:
    """What a slash command tells the REPL to do: show a message and/or stop."""

    message: RenderableType | None = None
    exit_session: bool = False


class SlashCommand(ABC):
    """One in-session command (``/name``), resolved and run by :func:`dispatch`."""

    name: str  # the keyword, without the leading slash (e.g. "clear")
    summary: str  # a one-line description shown by /help

    @abstractmethod
    def run(self, session: Session, argument: str) -> CommandResult:
        """Act on ``session``; ``argument`` is the (stripped) text after the name."""


_COMMANDS: dict[str, SlashCommand] = {}


def _register[C: SlashCommand](command_class: type[C]) -> type[C]:
    """Class decorator: instantiate the command and add it to the registry."""
    command = command_class()
    _COMMANDS[command.name] = command
    return command_class


def registered_commands() -> tuple[SlashCommand, ...]:
    """Every registered command, in registration order (for ``/help``)."""
    return tuple(_COMMANDS.values())


def dispatch(session: Session, line: str) -> CommandResult:
    """Resolve a ``/``-prefixed ``line`` to a command and run it.

    An unknown command yields a friendly message and never reaches the model.
    """
    name, _, argument = line[1:].partition(" ")
    command = _COMMANDS.get(name.strip().lower())
    if command is None:
        return CommandResult(
            message=Text(f"Unknown command: /{name.strip()} — type /help.", style=_ERROR_STYLE)
        )
    return command.run(session, argument.strip())


@_register
class HelpCommand(SlashCommand):
    name = "help"
    summary = "List the available commands."

    def run(self, session: Session, argument: str) -> CommandResult:
        body = Text()
        for command in registered_commands():
            body.append(f"/{command.name}", style="bold")
            body.append(f" — {command.summary}\n")
        return CommandResult(message=body)


@_register
class ModelCommand(SlashCommand):
    name = "model"
    summary = "Show the active provider and model."

    def run(self, session: Session, argument: str) -> CommandResult:
        if argument:
            return CommandResult(
                message=Text(
                    "Switching model mid-session isn't supported yet — "
                    "restart with --provider / --model.",
                    style=_WARN_STYLE,
                )
            )
        line = f"{session.provider_name} · {session.model} · effort {session.effort.value}"
        return CommandResult(message=Text(line, style=_INFO_STYLE))


@_register
class EffortCommand(SlashCommand):
    name = "effort"
    summary = "Show or set reasoning effort (off, low, medium, high, xhigh, max)."

    def run(self, session: Session, argument: str) -> CommandResult:
        if not argument:
            return CommandResult(
                message=Text(f"Reasoning effort: {session.effort.value}", style=_INFO_STYLE)
            )
        try:
            level = ReasoningEffort(argument.lower())
        except ValueError:
            levels = ", ".join(effort.value for effort in ReasoningEffort)
            return CommandResult(
                message=Text(
                    f"Unknown effort: {argument}. Choose from: {levels}.", style=_ERROR_STYLE
                )
            )
        session.set_effort(level)
        return CommandResult(
            message=Text(f"Reasoning effort set to {level.value}.", style=_NOTE_STYLE)
        )


@_register
class UsageCommand(SlashCommand):
    name = "usage"
    summary = "Show token usage so far this session."

    def run(self, session: Session, argument: str) -> CommandResult:
        usage = session.usage
        parts: list[str] = []
        context = format_context(session.context_budget)
        if context is not None:
            parts.append(context)
        parts.append(
            f"{SENT_GLYPH} {format_token_count(usage.input_tokens)} "
            f"{RECEIVED_GLYPH} {format_token_count(usage.output_tokens)} tokens"
        )
        return CommandResult(message=Text("Session usage: " + " · ".join(parts), style=_INFO_STYLE))


@_register
class ClearCommand(SlashCommand):
    name = "clear"
    summary = "Reset the conversation history."

    def run(self, session: Session, argument: str) -> CommandResult:
        session.clear()
        return CommandResult(message=Text("Conversation cleared.", style=_NOTE_STYLE))


@_register
class CompactCommand(SlashCommand):
    name = "compact"
    summary = "Summarise older turns to free up the context window."

    def run(self, session: Session, argument: str) -> CommandResult:
        event = session.compact()
        if event is None:
            return CommandResult(
                message=Text(
                    "Nothing to compact yet — the conversation is still short.", style=_NOTE_STYLE
                )
            )
        return CommandResult(message=format_compaction(event))


@_register
class ExitCommand(SlashCommand):
    name = "exit"
    summary = "End the session."

    def run(self, session: Session, argument: str) -> CommandResult:
        return CommandResult(message=Text("Goodbye.", style=_NOTE_STYLE), exit_session=True)

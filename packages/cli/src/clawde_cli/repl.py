"""The interactive REPL (#10): a multi-turn loop on top of a Session.

Running ``clawde`` with no prompt drops here. The loop reads a line, dispatches a
``/command`` (intercepted before the model, #11) or runs it as a turn, and keeps
going until Ctrl-D or ``/exit``. The line reader is injected so the loop is
testable offline; the default reads from prompt_toolkit (history, line editing).
A Ctrl-C raised while reading cancels the current line; during a turn the Session
catches it and the loop simply reads the next prompt.
"""

from __future__ import annotations

from collections.abc import Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import Input
from prompt_toolkit.output import Output
from rich.console import Console
from rich.text import Text

from clawde_cli.commands import dispatch
from clawde_cli.session import Session

PROMPT = "› "
_WELCOME_STYLE = "dim"

type LineReader = Callable[[], str]


def run_repl(session: Session, console: Console, read_line: LineReader) -> None:
    """Run the interactive loop until Ctrl-D or ``/exit``.

    ``read_line`` returns the next input line and raises ``EOFError`` on Ctrl-D.
    A ``/``-prefixed line is dispatched as a command (never sent to the model);
    anything else drives a turn. A ``KeyboardInterrupt`` raised while reading
    cancels that line and the loop reads the next.
    """
    console.print(_welcome())
    while True:
        try:
            line = read_line().strip()
        except EOFError:
            console.print()  # move past the half-typed prompt line after Ctrl-D
            break
        except KeyboardInterrupt:
            continue  # Ctrl-C at the prompt: abandon this line, read the next
        if not line:
            continue
        if line.startswith("/"):
            result = dispatch(session, line)
            if result.message is not None:
                console.print(result.message)
            if result.exit_session:
                break
            continue
        session.run_turn(line)


def interactive_line_reader(
    *, pt_input: Input | None = None, output: Output | None = None
) -> LineReader:
    """A line reader backed by prompt_toolkit (history, line editing, paste).

    ``pt_input`` / ``output`` are injected only by tests (a pipe + dummy output);
    in normal use they are ``None`` and prompt_toolkit binds the real terminal.
    """
    prompt_session: PromptSession[str] = PromptSession(
        history=InMemoryHistory(), input=pt_input, output=output
    )

    def read() -> str:
        return prompt_session.prompt(PROMPT)

    return read


def _welcome() -> Text:
    return Text(
        "clawde — interactive session. Type /help for commands, Ctrl-D or /exit to quit.",
        style=_WELCOME_STYLE,
    )

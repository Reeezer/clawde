"""The ``clawde`` command-line entry point.

``clawde "<prompt>"`` runs a single agent turn against Gemini (Phase 1's only
provider); ``--version`` prints the version. The interactive REPL and provider
selection arrive in later roadmap phases and will hang off this same app.
"""

from __future__ import annotations

import platform
from typing import Annotated

import typer
from clawde_core.config import get_settings
from clawde_core.loop import Agent, AgentError, Turn
from clawde_core.providers.base import ProviderError
from clawde_core.providers.gemini import DEFAULT_MODEL, GeminiProvider
from clawde_core.tools.bash import BashTool
from rich.console import Console

from clawde_cli import __version__

app = typer.Typer(
    name="clawde",
    help="clawde — a bring-your-own-model coding agent.",
    add_completion=False,
)
console = Console()


@app.callback(invoke_without_command=True)
def main(
    prompt: Annotated[
        str | None, typer.Argument(help="A task for the agent to carry out in one turn.")
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Gemini model id (default: gemini-2.5-flash).")
    ] = None,
    show_version: Annotated[
        bool, typer.Option("--version", help="Show the clawde version and exit.")
    ] = False,
) -> None:
    """clawde — a bring-your-own-model coding agent."""
    if show_version:
        console.print(f"clawde {__version__}")
        return
    if prompt is None:
        console.print(
            "[bold]clawde[/bold] — give me a task, e.g. "
            '[cyan]clawde "list the python files"[/cyan].\n'
            "The interactive REPL is on the roadmap."
        )
        return
    _run_turn(prompt, model)


def _run_turn(prompt: str, model: str | None) -> None:
    settings = get_settings()
    api_key = settings.providers.gemini.api_key
    if not api_key:
        console.print(
            "[red]No Gemini API key.[/red] Set CLAWDE_PROVIDERS__GEMINI__API_KEY in your .env."
        )
        raise typer.Exit(code=1)

    provider = GeminiProvider(
        api_key=api_key, model=model or settings.default_model or DEFAULT_MODEL
    )
    agent = Agent(provider, [BashTool()], system_prompt=_system_prompt())
    try:
        turn = agent.run_turn(prompt)
    except (ProviderError, AgentError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    _render(turn)


def _system_prompt() -> str:
    return (
        "You are clawde, a coding agent working in a terminal on "
        f"{platform.system()}. You have a `bash` tool that runs shell commands. "
        "Use it to inspect the project and accomplish the user's request, then "
        "reply with a concise final answer."
    )


def _render(turn: Turn) -> None:
    # markup=False: tool args and model text are data, not rich markup (they may
    # contain "[..]"). ASCII marker keeps output safe on cp1252 Windows consoles.
    for message in turn.messages:
        for call in message.tool_calls:
            console.print(f"-> {call.name}: {call.arguments}", style="dim", markup=False)
    console.print(turn.final_text, markup=False)
    console.print(f"({turn.usage.total} tokens used)", style="dim", markup=False)

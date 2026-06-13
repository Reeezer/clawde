"""The ``clawde`` command-line entry point.

Thin for now: a ``version`` command and a welcome callback. The interactive
agent REPL arrives in a later roadmap phase (see ``docs/ROADMAP.md``); it will
hang off this same Typer app.
"""

from __future__ import annotations

import typer
from rich.console import Console

from clawde_cli import __version__

app = typer.Typer(
    name="clawde",
    help="clawde — a bring-your-own-model coding agent.",
    add_completion=False,
)
console = Console()


@app.command()
def version() -> None:
    """Print the clawde version."""
    console.print(f"clawde {__version__}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """clawde — a bring-your-own-model coding agent."""
    if ctx.invoked_subcommand is None:
        console.print(
            "[bold]clawde[/bold] — the interactive REPL is on the roadmap.\n"
            "Run [cyan]clawde version[/cyan] or [cyan]clawde --help[/cyan]."
        )

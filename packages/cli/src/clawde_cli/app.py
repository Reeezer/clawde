"""The ``clawde`` command-line entry point.

``clawde "<prompt>"`` runs a single agent turn against the configured provider,
streaming the reply as it arrives. ``--provider`` / ``--model`` override the
choice for one run and ``--version`` prints the version. The provider is built
by the factory from settings (ADR-0003), so the CLI never names a concrete
backend; the interactive REPL arrives in a later roadmap phase.
"""

from __future__ import annotations

import mimetypes
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from clawde_core.loop import Agent, AgentError
from clawde_core.models import ImageContent, ToolCall
from clawde_core.providers.base import ProviderError
from clawde_core.providers.factory import build_provider
from clawde_core.registry import RegistryError
from clawde_core.tools.bash import BashTool
from rich.console import Console

from clawde_cli import __version__

app = typer.Typer(
    name="clawde",
    help="clawde — a bring-your-own-model coding agent.",
    add_completion=False,
)
console = Console()

_MAX_IMAGE_MB = 20
_MAX_IMAGE_BYTES = _MAX_IMAGE_MB * 1024 * 1024


@app.callback(invoke_without_command=True)
def main(
    prompt: Annotated[
        str | None, typer.Argument(help="A task for the agent to carry out in one turn.")
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider to use (e.g. anthropic, gemini, openai)."),
    ] = None,
    model: Annotated[
        str | None, typer.Option(help="Model id to use (default: the provider's own default).")
    ] = None,
    image: Annotated[
        list[Path] | None,
        typer.Option("--image", help="Attach an image file to the prompt (repeatable)."),
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
    _run_turn(prompt, provider, model, image or [])


def _run_turn(
    prompt: str, provider: str | None, model: str | None, image_paths: Sequence[Path]
) -> None:
    images = _load_images(image_paths)
    try:
        model_provider = build_provider(provider, model)
    except (ProviderError, RegistryError) as exc:
        _fail(str(exc))
    agent = Agent(model_provider, [BashTool()], system_prompt=_system_prompt())
    try:
        turn = agent.stream_turn(
            prompt, on_text=_emit_text, on_tool_call=_emit_tool_call, images=images
        )
    except (ProviderError, AgentError) as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print()  # end the streamed line
    console.print(f"({turn.usage.total} tokens used)", style="dim", markup=False)


def _load_images(paths: Sequence[Path]) -> tuple[ImageContent, ...]:
    """Read and validate ``--image`` paths into typed :class:`ImageContent`.

    Fails fast with a friendly message on a missing path, a non-image (or
    unrecognised) file, an unreadable file, or one above the size cap — so bad
    input never reaches the provider. Image bytes never touch the logs.
    """
    images: list[ImageContent] = []
    for path in paths:
        if not path.is_file():
            _fail(f"Image not found: {path}")
        mime, _ = mimetypes.guess_type(path.name)
        if mime is None or not mime.startswith("image/"):
            _fail(f"Not a recognised image file: {path}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            _fail(f"Could not read image {path}: {exc}")
        if len(data) > _MAX_IMAGE_BYTES:
            _fail(f"Image is too large (max {_MAX_IMAGE_MB} MB): {path}")
        images.append(ImageContent(mime_type=mime, data=data))
    return tuple(images)


def _fail(message: str) -> NoReturn:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(code=1)


def _emit_text(delta: str) -> None:
    console.print(delta, end="", markup=False, highlight=False)


def _emit_tool_call(call: ToolCall) -> None:
    # Break the streamed line, then show the call as data (not rich markup).
    console.print(f"\n-> {call.name}: {call.arguments}", style="dim", markup=False, highlight=False)


def _system_prompt() -> str:
    return (
        "You are clawde, a coding agent working in a terminal on "
        f"{platform.system()}. You have a `bash` tool that runs shell commands. "
        "Use it to inspect the project and accomplish the user's request, then "
        "reply with a concise final answer."
    )

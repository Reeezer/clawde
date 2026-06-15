"""The ``clawde`` command-line entry point.

``clawde "<prompt>"`` runs a single agent turn against the configured provider,
streaming the reply as it arrives; ``clawde`` with no prompt drops into an
interactive REPL instead (#10). ``--provider`` / ``--model`` override the choice
for one run and ``--version`` prints the version. The provider is built by the
factory from settings (ADR-0003), so the CLI never names a concrete backend.
"""

from __future__ import annotations

import mimetypes
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from clawde_core.models import ImageContent, ReasoningEffort
from clawde_core.providers.base import ProviderError
from clawde_core.registry import RegistryError

from clawde_cli import __version__
from clawde_cli.rendering import make_console
from clawde_cli.repl import interactive_line_reader, run_repl
from clawde_cli.session import build_session

app = typer.Typer(
    name="clawde",
    help="clawde — a bring-your-own-model coding agent.",
    add_completion=False,
)
console = make_console()

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
    effort: Annotated[
        ReasoningEffort | None,
        typer.Option("--effort", help="Reasoning effort: off, low, medium, high, xhigh, max."),
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
    _force_utf8(sys.stdout, sys.stderr)
    if show_version:
        console.print(f"clawde {__version__}")
        return
    if prompt is None:
        _run_repl(provider, model, effort)
        return
    _run_turn(prompt, provider, model, image or [], effort)


def _run_turn(
    prompt: str,
    provider: str | None,
    model: str | None,
    image_paths: Sequence[Path],
    effort: ReasoningEffort | None = None,
) -> None:
    images = _load_images(image_paths)
    try:
        session = build_session(console, provider=provider, model=model, effort=effort)
    except (ProviderError, RegistryError) as exc:
        _fail(str(exc))
    # The Session renders the reply and reports failure in its outcome (never
    # raising); one-shot turns it into a non-zero exit.
    if not session.run_turn(prompt, images).ok:
        raise typer.Exit(code=1)


def _run_repl(provider: str | None, model: str | None, effort: ReasoningEffort | None) -> None:
    try:
        session = build_session(console, provider=provider, model=model, effort=effort)
    except (ProviderError, RegistryError) as exc:
        _fail(str(exc))
    run_repl(session, console, interactive_line_reader())


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


def _force_utf8(*streams: object) -> None:
    """Make output UTF-8 so clawde's non-ASCII chrome (●, ✻, …) survives a
    redirected Windows console, which otherwise defaults to cp1252 and crashes.
    Streams that can't be reconfigured (e.g. test buffers) are left untouched.
    """
    for stream in streams:
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")

"""Enable ``python -m clawde_cli`` as an alias for the ``clawde`` command."""

from __future__ import annotations

from clawde_cli.app import app

if __name__ == "__main__":  # pragma: no cover
    app()

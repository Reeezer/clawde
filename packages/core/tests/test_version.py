from __future__ import annotations

import clawde_core


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(clawde_core.__version__, str)
    assert clawde_core.__version__

from __future__ import annotations

import pytest

from clawde_core.config import Settings
from clawde_core.registry import RegistryError
from clawde_core.tools.base import Tool
from clawde_core.tools.bash import BashTool
from clawde_core.tools.registry import TOOLS, build_tools, create_tool

_EXPECTED = ("bash", "edit", "glob", "grep", "read", "write")


def test_registry_holds_every_core_tool() -> None:
    assert TOOLS.keys() == _EXPECTED


def test_create_tool_builds_the_named_tool() -> None:
    tool = create_tool("bash")

    assert isinstance(tool, BashTool)
    assert tool.spec.name == "bash"


def test_create_tool_rejects_an_unknown_name() -> None:
    with pytest.raises(RegistryError, match="unknown key 'nope'"):
        create_tool("nope")


def test_build_tools_builds_the_full_set_from_settings() -> None:
    tools = build_tools(Settings())

    assert all(isinstance(tool, Tool) for tool in tools)
    assert tuple(tool.spec.name for tool in tools) == _EXPECTED

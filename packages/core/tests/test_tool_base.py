from __future__ import annotations

from collections.abc import Mapping

from clawde_core.models import ToolSpec
from clawde_core.tools.base import Tool


class _SchemaTool(Tool):
    """A fake tool requiring a string ``x``; records whether ``run`` was reached."""

    def __init__(self, parameters: dict[str, object]) -> None:
        self._parameters = parameters
        self.ran_with: dict[str, object] | None = None

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(name="fake", description="a fake tool", parameters=self._parameters)

    def run(self, arguments: Mapping[str, object]) -> str:
        self.ran_with = dict(arguments)
        return "ran"


_REQUIRES_X: dict[str, object] = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


def test_invoke_runs_when_arguments_are_valid() -> None:
    tool = _SchemaTool(_REQUIRES_X)

    result = tool.invoke({"x": "hello"})

    assert result == "ran"
    assert tool.ran_with == {"x": "hello"}


def test_invoke_rejects_missing_required_field_without_running() -> None:
    tool = _SchemaTool(_REQUIRES_X)

    result = tool.invoke({})

    assert result.startswith("Error:")
    assert "required" in result
    assert tool.ran_with is None  # run was never reached


def test_invoke_rejects_wrong_type_without_running() -> None:
    tool = _SchemaTool(_REQUIRES_X)

    result = tool.invoke({"x": 123})

    assert result.startswith("Error:")
    assert "not of type" in result
    assert tool.ran_with is None


def test_invoke_accepts_anything_under_an_empty_schema() -> None:
    tool = _SchemaTool({})

    result = tool.invoke({"whatever": 1})

    assert result == "ran"
    assert tool.ran_with == {"whatever": 1}

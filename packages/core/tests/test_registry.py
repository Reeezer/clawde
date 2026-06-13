from __future__ import annotations

import pytest

from clawde_core.registry import Registry, RegistryError


def test_register_and_create_builds_the_implementation() -> None:
    reg: Registry[str] = Registry("greeters")

    @reg.register("hi")
    def _build_hi() -> str:
        return "hello"

    assert reg.create("hi") == "hello"


def test_register_returns_the_builder_unchanged() -> None:
    reg: Registry[int] = Registry("nums")

    def build() -> int:
        return 7

    assert reg.register("seven")(build) is build


def test_duplicate_key_raises() -> None:
    reg: Registry[int] = Registry("nums")
    reg.register("one")(lambda: 1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("one")(lambda: 11)


def test_unknown_key_raises_and_lists_known_keys() -> None:
    reg: Registry[int] = Registry("nums")
    reg.register("one")(lambda: 1)
    with pytest.raises(RegistryError, match="unknown key 'two'"):
        reg.create("two")


def test_unknown_key_on_empty_registry_says_none() -> None:
    reg: Registry[int] = Registry("nums")
    with pytest.raises(RegistryError, match=r"known: \(none\)"):
        reg.create("two")


def test_keys_are_sorted() -> None:
    reg: Registry[int] = Registry("nums")
    reg.register("b")(lambda: 2)
    reg.register("a")(lambda: 1)
    assert reg.keys() == ("a", "b")


def test_contains() -> None:
    reg: Registry[int] = Registry("nums")
    reg.register("a")(lambda: 1)
    assert "a" in reg
    assert "z" not in reg

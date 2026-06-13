"""A tiny generic registry mapping a name to a builder for an implementation.

This is the backbone of clawde's swappable architecture: model providers and
tools each live behind one of these. Adding an implementation is "write the
class in its own file and register a builder" — nothing in the orchestrator
changes (see ADR-0003).
"""

from __future__ import annotations

from collections.abc import Callable

type Builder[T] = Callable[[], T]


class RegistryError(KeyError):
    """Raised when a key is missing, or when one is registered twice."""


class Registry[T]:
    """Maps string keys to zero-argument builders for ``T``.

    One instance per swappable stage (e.g. ``PROVIDERS``, ``TOOLS``).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._builders: dict[str, Builder[T]] = {}

    def register(self, key: str) -> Callable[[Builder[T]], Builder[T]]:
        """Decorator registering ``builder`` under ``key``; rejects duplicates."""

        def decorate(builder: Builder[T]) -> Builder[T]:
            if key in self._builders:
                raise RegistryError(f"{self._name}: '{key}' is already registered")
            self._builders[key] = builder
            return builder

        return decorate

    def create(self, key: str) -> T:
        """Build the implementation registered under ``key``."""
        try:
            builder = self._builders[key]
        except KeyError:
            known = ", ".join(self.keys()) or "(none)"
            raise RegistryError(f"{self._name}: unknown key '{key}'; known: {known}") from None
        return builder()

    def keys(self) -> tuple[str, ...]:
        """The registered keys, sorted."""
        return tuple(sorted(self._builders))

    def __contains__(self, key: object) -> bool:
        return key in self._builders

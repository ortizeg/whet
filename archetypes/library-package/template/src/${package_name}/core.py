"""Core registry primitives for ${project_name}.

A :class:`Registry` maps a stable string key to a factory (a class or a
function) that produces a value. It is the extension point downstream code uses
to turn configuration strings into objects without importing the concrete
implementation at the call site::

    from ${package_name} import Registry

    encoders: Registry[Encoder] = Registry("encoders")

    @encoders.register("identity")
    class IdentityEncoder:
        ...

    encoder = encoders.create("identity")

Logging uses Loguru and, per library convention, no sink is configured here.
Applications that import this package decide where the records go; until then
Loguru's own default handler applies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

from loguru import logger

__all__ = ["Registry", "RegistryError", "component_registry"]

T = TypeVar("T")
FactoryT = TypeVar("FactoryT", bound=Callable[..., Any])


class RegistryError(KeyError):
    """Raised when a registry key is unknown or already taken.

    Subclasses :class:`KeyError` so existing ``except KeyError`` handlers around
    dictionary-style lookups keep working.
    """


def normalize_key(key: str) -> str:
    """Normalize a registry key to its canonical form.

    Keys are compared case-insensitively with surrounding whitespace removed, so
    ``"ResNet50"``, ``"resnet50"`` and ``" resnet50 "`` all address the same
    entry. This makes keys safe to read straight out of a config file.

    Args:
        key: The raw key supplied by the caller.

    Returns:
        The lower-cased, stripped key.

    Raises:
        ValueError: If ``key`` is empty or only whitespace.
    """
    normalized = key.strip().lower()
    if not normalized:
        raise ValueError("registry keys must be non-empty strings")
    return normalized


class Registry(Generic[T]):
    """A name-keyed registry of factories that produce values of type ``T``.

    The registry stores callables rather than instances, so entries stay cheap
    to register at import time and are only constructed on demand.

    Args:
        name: Human-readable name used in log records and error messages.

    Raises:
        ValueError: If ``name`` is empty or only whitespace.

    Example:
        >>> greeters: Registry[str] = Registry("greeters")
        >>> _ = greeters.add("hello", lambda who: f"Hello, {who}!")
        >>> greeters.create("hello", "world")
        'Hello, world!'
    """

    def __init__(self, name: str) -> None:
        if not name.strip():
            raise ValueError("registry name must be a non-empty string")
        self._name = name.strip()
        self._factories: dict[str, Callable[..., T]] = {}

    @property
    def name(self) -> str:
        """The human-readable name of this registry."""
        return self._name

    def add(self, key: str, factory: Callable[..., T], *, override: bool = False) -> str:
        """Register ``factory`` under ``key``.

        Args:
            key: Key to register under; normalized via :func:`normalize_key`.
            factory: Callable returning a value of type ``T``.
            override: Allow replacing an existing entry. Defaults to ``False``
                so that duplicate registrations fail loudly instead of silently
                shadowing each other.

        Returns:
            The normalized key that was registered.

        Raises:
            RegistryError: If ``key`` is taken and ``override`` is ``False``.
            ValueError: If ``key`` is empty or ``factory`` is not callable.
        """
        normalized = normalize_key(key)
        if not callable(factory):
            raise ValueError(f"factory for {normalized!r} is not callable")
        if normalized in self._factories and not override:
            raise RegistryError(
                f"{normalized!r} is already registered in registry {self._name!r}; "
                "pass override=True to replace it"
            )
        self._factories[normalized] = factory
        logger.debug("Registered {!r} in registry {!r}", normalized, self._name)
        return normalized

    def register(self, key: str, *, override: bool = False) -> Callable[[FactoryT], FactoryT]:
        """Return a decorator that registers the decorated callable under ``key``.

        Args:
            key: Key to register under; normalized via :func:`normalize_key`.
            override: Allow replacing an existing entry.

        Returns:
            A decorator that registers and then returns its argument unchanged,
            so the decorated class or function stays directly importable.
        """

        def decorator(factory: FactoryT) -> FactoryT:
            self.add(key, factory, override=override)
            return factory

        return decorator

    def get(self, key: str) -> Callable[..., T]:
        """Look up the factory registered under ``key``.

        Args:
            key: Key to look up; normalized via :func:`normalize_key`.

        Returns:
            The registered factory.

        Raises:
            RegistryError: If no factory is registered under ``key``.
            ValueError: If ``key`` is empty.
        """
        normalized = normalize_key(key)
        try:
            return self._factories[normalized]
        except KeyError:
            known = ", ".join(self.names()) or "<empty>"
            raise RegistryError(
                f"{normalized!r} is not registered in registry {self._name!r}; known keys: {known}"
            ) from None

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """Look up ``key`` and call its factory.

        Args:
            key: Key to look up; normalized via :func:`normalize_key`.
            *args: Positional arguments forwarded to the factory.
            **kwargs: Keyword arguments forwarded to the factory.

        Returns:
            The value produced by the factory.

        Raises:
            RegistryError: If no factory is registered under ``key``.
        """
        factory = self.get(key)
        logger.debug("Creating {!r} from registry {!r}", normalize_key(key), self._name)
        return factory(*args, **kwargs)

    def unregister(self, key: str) -> None:
        """Remove the entry registered under ``key``.

        Args:
            key: Key to remove; normalized via :func:`normalize_key`.

        Raises:
            RegistryError: If no factory is registered under ``key``.
        """
        normalized = normalize_key(key)
        if self._factories.pop(normalized, None) is None:
            raise RegistryError(f"{normalized!r} is not registered in registry {self._name!r}")
        logger.debug("Unregistered {!r} from registry {!r}", normalized, self._name)

    def names(self) -> list[str]:
        """Return the registered keys in sorted order."""
        return sorted(self._factories)

    def clear(self) -> None:
        """Remove every entry. Primarily useful in tests."""
        self._factories.clear()

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` names a registered factory."""
        if not isinstance(key, str) or not key.strip():
            return False
        return normalize_key(key) in self._factories

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered keys in sorted order."""
        return iter(self.names())

    def __len__(self) -> int:
        """Return the number of registered factories."""
        return len(self._factories)

    def __repr__(self) -> str:
        """Return a debugging representation including the entry count."""
        return f"{type(self).__name__}(name={self._name!r}, entries={len(self._factories)})"


component_registry: Registry[Any] = Registry("${project_slug}.components")
"""Package-level registry that downstream code and plugins can populate."""

"""Shared fixtures for the ${project_name} test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from ${package_name}.config import ENV_PREFIX
from ${package_name}.core import Registry, component_registry


@pytest.fixture
def greeters() -> Registry[str]:
    """An empty registry that produces strings."""
    return Registry("greeters")


@pytest.fixture
def populated_greeters(greeters: Registry[str]) -> Registry[str]:
    """A registry pre-populated with two factories."""
    greeters.add("hello", lambda who: f"Hello, {who}!")
    greeters.add("howdy", lambda who: f"Howdy, {who}!")
    return greeters


@pytest.fixture
def clean_components() -> Iterator[Registry[Any]]:
    """The package-level registry, emptied before and after the test."""
    component_registry.clear()
    yield component_registry
    component_registry.clear()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every ``ENV_PREFIX``-scoped variable from the process environment."""
    for key in [k for k in os.environ if k.startswith(ENV_PREFIX)]:
        monkeypatch.delenv(key, raising=False)
    yield

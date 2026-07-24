"""Tests for the ${project_name} registry."""

from __future__ import annotations

from typing import Any

import pytest

from ${package_name} import Registry, RegistryError
from ${package_name}.core import normalize_key


def test_registry_requires_a_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Registry("   ")


def test_registry_name_is_stripped() -> None:
    assert Registry("  models  ").name == "models"


def test_add_returns_normalized_key(greeters: Registry[str]) -> None:
    assert greeters.add("  Hello  ", lambda who: f"Hi {who}") == "hello"


def test_create_invokes_the_factory(populated_greeters: Registry[str]) -> None:
    assert populated_greeters.create("hello", "world") == "Hello, world!"
    assert populated_greeters.create("howdy", who="partner") == "Howdy, partner!"


def test_lookup_is_case_and_whitespace_insensitive(populated_greeters: Registry[str]) -> None:
    assert populated_greeters.create(" HELLO ", "world") == "Hello, world!"


def test_duplicate_registration_is_rejected(populated_greeters: Registry[str]) -> None:
    with pytest.raises(RegistryError, match="already registered"):
        populated_greeters.add("hello", lambda who: f"nope {who}")


def test_override_replaces_the_entry(populated_greeters: Registry[str]) -> None:
    populated_greeters.add("hello", lambda who: f"Yo, {who}!", override=True)
    assert populated_greeters.create("hello", "world") == "Yo, world!"
    assert len(populated_greeters) == 2


def test_missing_key_lists_known_keys(populated_greeters: Registry[str]) -> None:
    with pytest.raises(RegistryError, match="hello, howdy"):
        populated_greeters.get("nope")


def test_missing_key_on_empty_registry(greeters: Registry[str]) -> None:
    with pytest.raises(RegistryError, match="<empty>"):
        greeters.get("nope")


def test_registry_error_is_a_key_error(greeters: Registry[str]) -> None:
    with pytest.raises(KeyError):
        greeters.get("nope")


def test_non_callable_factory_is_rejected(greeters: Registry[Any]) -> None:
    with pytest.raises(ValueError, match="not callable"):
        greeters.add("oops", "not a factory")  # type: ignore[arg-type]


@pytest.mark.parametrize("key", ["", "   ", "\t\n"])
def test_empty_keys_are_rejected(greeters: Registry[str], key: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        greeters.add(key, lambda who: str(who))


def test_register_decorator_returns_the_original_callable(greeters: Registry[str]) -> None:
    @greeters.register("shout")
    def shout(who: str) -> str:
        return f"HEY {who.upper()}"

    assert shout("world") == "HEY WORLD"
    assert greeters.create("shout", "world") == "HEY WORLD"


def test_register_decorator_honours_override(greeters: Registry[str]) -> None:
    greeters.add("shout", lambda who: "old")

    @greeters.register("shout", override=True)
    def shout(who: str) -> str:
        return "new"

    assert greeters.create("shout", "world") == "new"


def test_unregister_removes_the_entry(populated_greeters: Registry[str]) -> None:
    populated_greeters.unregister("HELLO")
    assert "hello" not in populated_greeters
    assert populated_greeters.names() == ["howdy"]


def test_unregister_unknown_key_raises(greeters: Registry[str]) -> None:
    with pytest.raises(RegistryError, match="not registered"):
        greeters.unregister("nope")


def test_container_protocol(populated_greeters: Registry[str]) -> None:
    assert len(populated_greeters) == 2
    assert "hello" in populated_greeters
    assert "Hello" in populated_greeters
    assert "missing" not in populated_greeters
    assert 42 not in populated_greeters
    assert "" not in populated_greeters
    assert list(populated_greeters) == ["hello", "howdy"]


def test_clear_empties_the_registry(populated_greeters: Registry[str]) -> None:
    populated_greeters.clear()
    assert len(populated_greeters) == 0
    assert populated_greeters.names() == []


def test_repr_reports_name_and_size(populated_greeters: Registry[str]) -> None:
    assert repr(populated_greeters) == "Registry(name='greeters', entries=2)"


def test_component_registry_is_usable(clean_components: Registry[Any]) -> None:
    @clean_components.register("demo")
    class Demo:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    instance = clean_components.create("demo", value=7)
    assert isinstance(instance, Demo)
    assert instance.value == 7
    assert clean_components.names() == ["demo"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Hello", "hello"), ("  MiXeD  ", "mixed"), ("a-b_c", "a-b_c")],
)
def test_normalize_key(raw: str, expected: str) -> None:
    assert normalize_key(raw) == expected

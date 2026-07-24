"""Tests for the ${project_name} configuration model."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ${package_name} import ENV_PREFIX, LibraryConfig


def test_defaults() -> None:
    config = LibraryConfig()
    assert config.log_level == "INFO"
    assert config.strict is True
    assert config.max_items == 1024


def test_config_is_frozen() -> None:
    config = LibraryConfig()
    with pytest.raises(ValidationError):
        config.max_items = 1


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LibraryConfig.model_validate({"log_levl": "DEBUG"})


@pytest.mark.parametrize("level", ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"])
def test_valid_log_levels(level: str) -> None:
    assert LibraryConfig.model_validate({"log_level": level}).log_level == level


def test_invalid_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        LibraryConfig.model_validate({"log_level": "LOUD"})


def test_max_items_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="greater_than_equal"):
        LibraryConfig.model_validate({"max_items": 0})


def test_env_prefix_is_derived_from_the_package_name() -> None:
    assert ENV_PREFIX.isupper()
    assert ENV_PREFIX.endswith("_")


def test_from_env_uses_defaults_when_unset(clean_env: None) -> None:
    assert LibraryConfig.from_env() == LibraryConfig()


def test_from_env_reads_and_coerces_values() -> None:
    env = {
        f"{ENV_PREFIX}LOG_LEVEL": "DEBUG",
        f"{ENV_PREFIX}STRICT": "false",
        f"{ENV_PREFIX}MAX_ITEMS": "16",
        "UNRELATED": "ignored",
    }
    config = LibraryConfig.from_env(env)
    assert config.log_level == "DEBUG"
    assert config.strict is False
    assert config.max_items == 16


def test_from_env_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        LibraryConfig.from_env({f"{ENV_PREFIX}MAX_ITEMS": "not-a-number"})


def test_from_env_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_ITEMS", "7")
    assert LibraryConfig.from_env().max_items == 7


def test_env_var_names_covers_every_field() -> None:
    names = LibraryConfig().env_var_names()
    assert set(names) == set(LibraryConfig.model_fields)
    assert names["max_items"] == f"{ENV_PREFIX}MAX_ITEMS"

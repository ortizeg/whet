"""Tests for the ${project_name} command-line interface."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ${package_name} import __version__
from ${package_name}.cli import build_parser, main, run
from ${package_name}.config import ENV_PREFIX
from ${package_name}.core import Registry


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    """Parse the JSON payload the CLI wrote to stdout."""
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    return payload


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_flag_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_info_reports_version_and_config(
    clean_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["info"]) == 0
    payload = _stdout_json(capsys)
    assert payload["version"] == __version__
    assert payload["config"]["log_level"] == "INFO"
    assert payload["env_vars"]["strict"] == f"{ENV_PREFIX}STRICT"


def test_info_honours_the_log_level_override(
    clean_env: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--log-level", "ERROR", "info"]) == 0
    assert _stdout_json(capsys)["config"]["log_level"] == "ERROR"


def test_info_reads_configuration_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(f"{ENV_PREFIX}MAX_ITEMS", "12")
    assert main(["info"]) == 0
    assert _stdout_json(capsys)["config"]["max_items"] == 12


def test_components_lists_registered_keys(
    clean_env: None,
    clean_components: Registry[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    clean_components.add("beta", lambda: "b")
    clean_components.add("alpha", lambda: "a")
    assert main(["components"]) == 0
    assert _stdout_json(capsys)["components"] == ["alpha", "beta"]


def test_components_on_an_empty_registry(
    clean_env: None,
    clean_components: Registry[Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["components"]) == 0
    assert _stdout_json(capsys)["components"] == []


def test_invalid_log_level_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--log-level", "LOUD", "info"])


def test_run_exits_with_the_return_code_of_main(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["${project_slug}", "info"])
    with pytest.raises(SystemExit) as excinfo:
        run()
    assert excinfo.value.code == 0
    assert _stdout_json(capsys)["package"] == "${package_name}"

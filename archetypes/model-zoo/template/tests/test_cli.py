"""End-to-end exercise of ``python -m ${package_name}``."""

from __future__ import annotations

from pathlib import Path

import pytest

from ${package_name}.__main__ import main


@pytest.fixture
def cli(registry_dir: Path) -> list[str]:
    return ["--registry", str(registry_dir)]


def test_list_exits_zero(cli: list[str]) -> None:
    assert main([*cli, "list"]) == 0


def test_list_with_no_matches_exits_one(cli: list[str]) -> None:
    assert main([*cli, "list", "--task", "segmentation"]) == 1


def test_show_known_model(cli: list[str]) -> None:
    assert main([*cli, "show", "resnet50"]) == 0


def test_show_unknown_model_exits_two(cli: list[str]) -> None:
    assert main([*cli, "show", "no-such-model"]) == 2


def test_missing_registry_exits_two(tmp_path: Path) -> None:
    assert main(["--registry", str(tmp_path / "nope"), "list"]) == 2


def test_verify_detects_a_bad_file(cli: list[str], tmp_path: Path) -> None:
    impostor = tmp_path / "resnet50-v1.pth"
    impostor.write_bytes(b"not the real weights")
    assert main([*cli, "verify", "resnet50", str(impostor)]) == 2

"""Tests for the Pydantic experiment configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ${package_name}.config import PATHS, ExperimentConfig, Paths


def test_config_defaults_are_valid() -> None:
    cfg = ExperimentConfig(name="baseline")
    assert cfg.seed == 42
    assert cfg.stage == "explore"
    assert cfg.test_fraction == pytest.approx(0.15)


def test_config_is_frozen(config: ExperimentConfig) -> None:
    with pytest.raises(ValidationError):
        config.seed = 1


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(name="baseline", learninig_rate=0.1)


@pytest.mark.parametrize("bad_name", ["", "Has Spaces", "UPPER", "-leading-dash"])
def test_config_rejects_bad_names(bad_name: str) -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(name=bad_name)


def test_config_rejects_negative_seed() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig(name="baseline", seed=-1)


def test_config_rejects_splits_leaving_no_test_set() -> None:
    with pytest.raises(ValidationError, match="leave a test split"):
        ExperimentConfig(name="baseline", train_fraction=0.9, val_fraction=0.2)


def test_variant_returns_modified_copy(config: ExperimentConfig) -> None:
    ablation = config.variant(seed=99, stage="ablation")
    assert ablation.seed == 99
    assert ablation.stage == "ablation"
    assert config.seed == 7, "original config must be untouched"


def test_variant_still_validates(config: ExperimentConfig) -> None:
    with pytest.raises(ValidationError):
        config.variant(n_classes=0)


def test_slug_and_figure_path(config: ExperimentConfig) -> None:
    assert config.slug == "unit-test-seed7"
    assert config.figure_path("scatter").name == "unit-test-seed7-scatter.png"
    assert config.figure_path("scatter").parent == PATHS.figures


def test_paths_are_rooted_and_nested() -> None:
    paths = Paths()
    assert paths.raw.parent == paths.data
    assert paths.figures.parent == paths.outputs
    assert paths.notebooks == paths.root / "notebooks"


def test_paths_ensure_creates_directories(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    paths.ensure()
    assert paths.raw.is_dir()
    assert paths.processed.is_dir()
    assert paths.figures.is_dir()
    assert paths.reports.is_dir()

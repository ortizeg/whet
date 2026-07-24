"""Tests for Hydra composition and the training entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ${package_name}.data import DataConfig, ImageDataModule
from ${package_name}.model import Classifier, ModelConfig
from ${package_name}.train import ExperimentConfig, TrainerConfig, build_trainer


def _compose(configs_dir: Path, overrides: list[str]) -> ExperimentConfig:
    with initialize_config_dir(version_base=None, config_dir=str(configs_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    return ExperimentConfig.from_dictconfig(cfg)


def test_default_config_composes(configs_dir: Path) -> None:
    """The default config tree validates into a typed experiment config."""
    config = _compose(configs_dir, [])

    assert config.seed == 42
    assert config.model.backbone == "resnet18"
    assert config.data.batch_size == 32
    assert config.trainer.max_epochs == 50


def test_cli_overrides_are_applied(configs_dir: Path) -> None:
    """Documented CLI overrides actually change the composed config."""
    config = _compose(configs_dir, ["model=resnet50", "trainer.max_epochs=5"])

    assert config.model.backbone == "resnet50"
    assert config.trainer.max_epochs == 5


def test_debug_trainer_group(configs_dir: Path) -> None:
    """`trainer=debug` selects the single-batch smoke configuration."""
    config = _compose(configs_dir, ["trainer=debug"])

    assert config.trainer.fast_dev_run is True
    assert config.trainer.max_epochs == 1
    assert config.trainer.accelerator == "cpu"


def test_invalid_override_is_rejected(configs_dir: Path) -> None:
    """Out-of-range values are caught by Pydantic before training starts."""
    with pytest.raises(ValueError, match="max_epochs"):
        _compose(configs_dir, ["trainer.max_epochs=0"])


def test_fast_dev_run_completes(
    model_config: ModelConfig,
    data_config: DataConfig,
) -> None:
    """End-to-end smoke run: a real training and validation step execute."""
    trainer = build_trainer(
        TrainerConfig(
            fast_dev_run=True,
            accelerator="cpu",
            devices=1,
            precision="32-true",
            max_epochs=1,
            log_every_n_steps=1,
        )
    )
    trainer.fit(
        Classifier(config=model_config),
        datamodule=ImageDataModule(config=data_config),
    )

    assert trainer.state.finished

"""Training entry point for ${project_name}.

Configuration is composed by Hydra from the project's ``configs/`` tree and then
validated with Pydantic before anything expensive runs:

    python -m ${package_name}.train
    python -m ${package_name}.train model=resnet50 trainer.max_epochs=50
    python -m ${package_name}.train trainer=debug
"""

from __future__ import annotations

from typing import Any

import hydra
import lightning as L
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from .data import DataConfig, ImageDataModule
from .model import Classifier, ModelConfig

# Relative to this file: src/${package_name}/ -> <project root>/configs
CONFIG_PATH = "../../configs"


class TrainerConfig(BaseModel, frozen=True):
    """Lightning ``Trainer`` configuration."""

    max_epochs: int = Field(default=50, ge=1)
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32-true"
    log_every_n_steps: int = Field(default=10, ge=1)
    fast_dev_run: bool = False
    limit_train_batches: float | None = None
    limit_val_batches: float | None = None


class ExperimentConfig(BaseModel, frozen=True):
    """Fully validated experiment configuration."""

    seed: int = Field(default=42, ge=0)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> ExperimentConfig:
        """Validate a composed Hydra config into a typed experiment config."""
        raw: Any = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError("Composed Hydra config must resolve to a mapping")
        return cls.model_validate(raw)


def build_trainer(config: TrainerConfig) -> L.Trainer:
    """Instantiate a Lightning ``Trainer`` from validated settings."""
    return L.Trainer(**config.model_dump())


def run(config: ExperimentConfig) -> L.Trainer:
    """Run training for a validated experiment configuration."""
    L.seed_everything(config.seed, workers=True)
    logger.info(
        "Starting training | backbone={} epochs={} lr={}",
        config.model.backbone,
        config.trainer.max_epochs,
        config.model.learning_rate,
    )

    model = Classifier(config=config.model)
    datamodule = ImageDataModule(config=config.data)
    trainer = build_trainer(config.trainer)

    trainer.fit(model, datamodule=datamodule)
    logger.info("Training complete")
    return trainer


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    logger.info("Resolved configuration:\n{}", OmegaConf.to_yaml(cfg))
    run(ExperimentConfig.from_dictconfig(cfg))


if __name__ == "__main__":
    main()

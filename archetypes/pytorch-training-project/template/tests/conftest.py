"""Shared pytest fixtures for ${project_name}."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from ${package_name}.data import DataConfig, ImageDataModule
from ${package_name}.model import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    """Absolute path to the Hydra config tree."""
    return PROJECT_ROOT / "configs"


@pytest.fixture
def model_config() -> ModelConfig:
    """Tiny model config; `pretrained=False` keeps tests offline and fast."""
    return ModelConfig(num_classes=4, pretrained=False)


@pytest.fixture
def data_config() -> DataConfig:
    """Tiny data config that runs in-process on CPU."""
    return DataConfig(
        batch_size=2,
        num_workers=0,
        num_classes=4,
        image_size=32,
        train_size=4,
        val_size=2,
    )


@pytest.fixture
def datamodule(data_config: DataConfig) -> ImageDataModule:
    """A datamodule with its `fit` datasets already built."""
    module = ImageDataModule(config=data_config)
    module.setup("fit")
    return module


@pytest.fixture
def image_batch(data_config: DataConfig) -> tuple[Tensor, Tensor]:
    """An `(images, labels)` batch matching the dataloader contract."""
    images = torch.randn(2, 3, data_config.image_size, data_config.image_size)
    labels = torch.randint(0, data_config.num_classes, (2,))
    return images, labels

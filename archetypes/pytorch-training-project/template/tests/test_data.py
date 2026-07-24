"""Tests for the data module."""

from __future__ import annotations

import pytest
import torch

from ${package_name}.data import DataConfig, ImageDataModule


def test_dataloader_yields_image_label_tuples(
    datamodule: ImageDataModule,
    data_config: DataConfig,
) -> None:
    """Batches unpack as `(images, labels)` — the contract the model relies on."""
    batch = next(iter(datamodule.train_dataloader()))

    images, labels = batch
    assert images.shape == (
        data_config.batch_size,
        3,
        data_config.image_size,
        data_config.image_size,
    )
    assert images.dtype == torch.float32
    assert labels.shape == (data_config.batch_size,)
    assert labels.dtype == torch.int64
    assert int(labels.max()) < data_config.num_classes


def test_val_dataloader_is_built(datamodule: ImageDataModule) -> None:
    """Validation batches follow the same tuple contract."""
    images, labels = next(iter(datamodule.val_dataloader()))
    assert images.ndim == 4
    assert labels.ndim == 1


def test_dataloader_requires_setup(data_config: DataConfig) -> None:
    """Requesting a dataloader before `setup()` fails loudly, not silently."""
    module = ImageDataModule(config=data_config)
    with pytest.raises(RuntimeError):
        module.train_dataloader()


def test_config_rejects_invalid_values() -> None:
    """Pydantic validation catches bad configuration at load time."""
    with pytest.raises(ValueError, match="batch_size"):
        DataConfig(batch_size=0)

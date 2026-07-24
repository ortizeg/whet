"""Tests for the model module."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from ${package_name}.model import Classifier, ModelConfig


def test_model_forward_shape(model_config: ModelConfig) -> None:
    """The model produces one logit per class."""
    model = Classifier(config=model_config)
    model.eval()

    images = torch.randn(2, 3, 32, 32)
    output = model(images)
    assert output.shape == (2, model_config.num_classes)


def test_training_step_accepts_tuple_batch(
    model_config: ModelConfig,
    image_batch: tuple[Tensor, Tensor],
) -> None:
    """`training_step` unpacks `(images, labels)` and returns a scalar loss."""
    model = Classifier(config=model_config)
    loss = model.training_step(image_batch, batch_idx=0)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_validation_step_accepts_tuple_batch(
    model_config: ModelConfig,
    image_batch: tuple[Tensor, Tensor],
) -> None:
    """`validation_step` uses the same tuple contract as training."""
    model = Classifier(config=model_config)
    assert model.validation_step(image_batch, batch_idx=0) is None


def test_backbone_is_validated() -> None:
    """An unknown backbone name fails fast with a clear error."""
    with pytest.raises(ValueError, match="backbone"):
        Classifier(config=ModelConfig(backbone="not_a_real_backbone", pretrained=False))

"""LightningModule for ${project_name}."""

from __future__ import annotations

from typing import cast

import lightning as L
import torch
import torchmetrics
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from pydantic import BaseModel, Field
from torch import Tensor, nn

from .data import Batch


class ModelConfig(BaseModel, frozen=True):
    """Model configuration."""

    num_classes: int = Field(default=10, ge=2)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    backbone: str = "resnet18"
    pretrained: bool = True


class Classifier(L.LightningModule):
    """Image classification model.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config

        self.backbone, feature_dim = self._build_backbone()
        self.head = nn.Linear(feature_dim, config.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.num_classes,
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=config.num_classes,
        )

    def _build_backbone(self) -> tuple[nn.Module, int]:
        """Build the backbone network and report its feature dimension."""
        from torchvision import models

        builder = getattr(models, self.config.backbone, None)
        if builder is None:
            raise ValueError(f"Unknown torchvision backbone: {self.config.backbone}")

        weights = "DEFAULT" if self.config.pretrained else None
        model = builder(weights=weights)
        head = getattr(model, "fc", None)
        if not isinstance(head, nn.Linear):
            raise ValueError(f"Backbone {self.config.backbone!r} has no `fc` layer to replace")

        feature_dim = int(head.in_features)
        # Drop the ImageNet classification head; `self.head` replaces it.
        model.fc = nn.Identity()
        return cast(nn.Module, model), feature_dim

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass over a batch of images."""
        features = self.backbone(x)
        return cast(Tensor, self.head(features))

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        """Run one training step on an ``(images, labels)`` batch."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.train_acc(logits, labels)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return cast(Tensor, loss)

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Run one validation step on an ``(images, labels)`` batch."""
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        self.val_acc(logits, labels)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Build the optimizer and LR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
        )
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

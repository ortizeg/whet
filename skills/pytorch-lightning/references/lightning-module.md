# LightningModule Design

The full reference LightningModule for image classification, plus the rules that govern how one should be written.

## Contents

- [Standard LightningModule Structure](#standard-lightningmodule-structure)
- [Key Rules for LightningModule](#key-rules-for-lightningmodule)

## Standard LightningModule Structure

Every model follows this structure: the `LightningModule` encapsulates architecture, loss, optimizer config, and step logic.

```python
"""LightningModule for image classification."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor


class ImageClassifier(L.LightningModule):
    """Image classification model using Lightning."""

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1e-3,
        backbone: str = "resnet50",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = self._build_backbone(backbone, num_classes, pretrained)
        self.loss_fn = nn.CrossEntropyLoss()

        # Metrics - one instance per phase to avoid cross-contamination
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _build_backbone(
        self, backbone: str, num_classes: int, pretrained: bool
    ) -> nn.Module:
        """Build the backbone network."""
        import torchvision.models as models

        weights = "IMAGENET1K_V2" if pretrained else None
        model = getattr(models, backbone)(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass - only used for inference."""
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        """Shared computation for train/val/test steps."""
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Validation step."""
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Test step."""
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }
```

## Key Rules for LightningModule

1. **Always call `self.save_hyperparameters()`** in `__init__` -- this enables automatic checkpoint loading and logging.
2. **Create separate metric instances** for train, val, and test to avoid state leakage between phases.
3. **Use `_shared_step`** to avoid duplicating forward logic across training_step, validation_step, and test_step.
4. **Log with namespaced keys** like `train/loss`, `val/acc` -- never use flat names like `loss` or `accuracy`.
5. **Return loss from `training_step`** -- Lightning uses it for backpropagation. Do not return anything from validation_step or test_step.
6. **Type all methods** with proper return annotations.

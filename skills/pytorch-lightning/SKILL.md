---
name: pytorch-lightning
description: >
  Use this skill when building a training pipeline with PyTorch Lightning —
  LightningModule and LightningDataModule design, Trainer configuration, callbacks,
  logger integration, distributed DDP/FSDP training, and checkpoint management. Reach for
  it any time you'd otherwise hand-write a raw PyTorch training loop with manual
  epoch/optimizer/device handling, even if the user just says "train this model". For the
  pretrained models and datasets you feed in see huggingface; for the logging backends see
  wandb, tensorboard, or mlflow.
---

# PyTorch Lightning Skill

Lightning separates research code (the model) from engineering code (training loops,
distributed training, logging). Use Lightning as the standard training abstraction; never
write raw training loops. This page is the index — the minimal module-plus-Trainer shape is
inline; the full patterns live in `references/` and should be read only when needed.

## The Essential Core

A `LightningModule` owning architecture, loss, steps, and optimizer, driven by a `Trainer`.

```python
"""Minimal Lightning training pipeline."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor


class ImageClassifier(L.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()  # required: enables checkpoint round-trip
        self.model = build_backbone(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        # One metric instance per phase — never share across train/val/test.
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        images, labels = batch
        logits = self(images)
        return self.loss_fn(logits, labels), torch.argmax(logits, dim=1), labels

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss  # Lightning backprops this

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}


L.seed_everything(42, workers=True)
trainer = L.Trainer(
    max_epochs=100,
    accelerator="auto",   # auto-detect CPU/GPU/TPU
    devices="auto",
    strategy="auto",      # auto-select DDP/FSDP when multi-device
    precision="16-mixed",
    deterministic=True,
    gradient_clip_val=1.0,
)
trainer.fit(model, datamodule=datamodule)
trainer.test(model, datamodule=datamodule, ckpt_path="best")
```

Data always arrives through a `LightningDataModule` (`setup(stage)` builds datasets,
`train_dataloader()` / `val_dataloader()` return them) — never loose DataLoaders in the
training script.

## Conventions

- Call `self.save_hyperparameters()` in every `__init__`; without it, `load_from_checkpoint` cannot rebuild the model.
- Separate `torchmetrics` instances per phase; they handle DDP sync for you.
- Factor forward/loss into a `_shared_step` used by train, val, and test steps.
- Namespace log keys: `train/loss`, `val/acc` — never flat `loss` or `accuracy`.
- Return the loss from `training_step`; return nothing from `validation_step` / `test_step`.
- Build datasets in `setup(stage)`, download data in `prepare_data()` (rank 0 only).
- `persistent_workers=True` with `num_workers > 0`, `pin_memory=True` on GPU, `drop_last=True` for training.
- Train and val transforms must differ — validation gets no random augmentation.
- Seed with `L.seed_everything(seed, workers=True)` and set `deterministic=True` for reproducible runs.
- Type every method with real return annotations.

## Anti-Patterns to Avoid

1. **Never write raw training loops** -- always use Lightning Trainer.
2. **Never call `.cuda()` or `.to(device)`** -- Lightning handles device placement.
3. **Never call `optimizer.zero_grad()` or `optimizer.step()`** -- Lightning handles this.
4. **Never call `loss.backward()`** -- Lightning handles backpropagation.
5. **Never use `model.train()` or `model.eval()`** -- Lightning manages train/eval mode.
6. **Never manually sync metrics in DDP** -- use `torchmetrics` which handles sync automatically.
7. **Never put data downloads in `setup()`** -- use `prepare_data()` which runs on rank 0 only.

## Deep dives

- `references/lightning-module.md` — read when writing or reviewing a LightningModule: full reference implementation, backbone construction, `configure_optimizers` with a scheduler, and the module rules.
- `references/datamodule.md` — read when building a LightningDataModule: transforms, `setup(stage)`, dataloader construction, and the DataModule rules.
- `references/trainer-and-callbacks.md` — read when configuring the Trainer or wiring callbacks: full training script, EarlyStopping/LearningRateMonitor setup, and writing a custom `Callback`.
- `references/distributed-training.md` — read when scaling to multiple GPUs or sharding a large model with DDP or FSDP.
- `references/checkpointing-and-logging.md` — read when configuring `ModelCheckpoint`, restoring weights, attaching multiple loggers, or logging images and artifacts.
- `references/testing-lightning-code.md` — read when writing tests for a LightningModule: `fast_dev_run` smoke tests and checkpoint round-trip checks.

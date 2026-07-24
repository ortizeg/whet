# Trainer Configuration and Callbacks

The full training script, Trainer arguments, built-in callbacks, and how to write a custom callback.

## Contents

- [Trainer Configuration](#trainer-configuration)
- [Custom Callback Pattern](#custom-callback-pattern)
- [Frequently Used Built-in Callbacks](#frequently-used-built-in-callbacks)

## Trainer Configuration

```python
"""Training script using Lightning Trainer."""

from __future__ import annotations

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger


def train() -> None:
    """Run training pipeline."""
    L.seed_everything(42, workers=True)

    datamodule = ImageClassificationDataModule(
        data_dir="data/imagenet", batch_size=64, num_workers=8
    )
    model = ImageClassifier(
        num_classes=1000, learning_rate=1e-3, backbone="resnet50", pretrained=True
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-{val/acc:.3f}",
            monitor="val/acc",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(monitor="val/loss", patience=10, mode="min", verbose=True),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]
    logger = WandbLogger(project="image-classification", name="resnet50-baseline", log_model=True)

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=50,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    train()
```

Key Trainer args: `accelerator`/`devices`/`strategy="auto"` (auto-detect hardware and DDP/FSDP), `precision="16-mixed"` (mixed precision), `deterministic=True` (reproducibility), `gradient_clip_val=1.0` (prevent gradient explosion).

## Custom Callback Pattern

```python
"""Custom callbacks for training monitoring."""

from __future__ import annotations

from typing import Any

import lightning as L
from lightning.pytorch.callbacks import Callback


class ImageLoggingCallback(Callback):
    """Log sample predictions as images to the logger."""

    def __init__(self, num_samples: int = 8) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log predictions on first batch of each validation epoch."""
        if batch_idx != 0:
            return

        images, labels = batch
        images = images[: self.num_samples]
        labels = labels[: self.num_samples]

        with torch.no_grad():
            logits = pl_module(images)
            preds = torch.argmax(logits, dim=1)

        # Log to wandb if available
        if hasattr(trainer.logger, "experiment"):
            import wandb

            trainer.logger.experiment.log({
                "val/predictions": [
                    wandb.Image(img, caption=f"pred={p}, true={t}")
                    for img, p, t in zip(images, preds, labels)
                ]
            })
```

## Frequently Used Built-in Callbacks

```python
from lightning.pytorch.callbacks import (
    ModelCheckpoint,      # Save best/last checkpoints
    EarlyStopping,        # Stop when metric plateaus
    LearningRateMonitor,  # Log learning rate to logger
    RichProgressBar,      # Better terminal progress bars
    StochasticWeightAveraging,  # SWA for better generalization
    GradientAccumulationScheduler,  # Variable accumulation
    ModelSummary,         # Print model architecture summary
)
```

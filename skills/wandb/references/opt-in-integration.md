# W&B Setup, Opt-in Wiring, and Framework Integration

Scope: installing and authenticating W&B, structuring projects, making tracking optional, wiring it into PyTorch Lightning, and running offline.

## Contents

- [Installation](#installation)
- [Authentication](#authentication)
- [Project Structure](#project-structure)
- [Using Pydantic Config with W&B](#using-pydantic-config-with-wb)
- [Opt-in Pattern](#opt-in-pattern)
- [Integration with PyTorch Lightning](#integration-with-pytorch-lightning)
- [Offline Mode](#offline-mode)

## Installation

```bash
# Using pip
pip install wandb

# Using pixi
pixi add wandb --feature experiment-tracking
```

## Authentication

```bash
# Login with your API key (one-time setup)
wandb login

# Or set the environment variable
export WANDB_API_KEY=your_api_key_here

# For CI/CD environments, use secrets
# GitHub Actions: ${{ secrets.WANDB_API_KEY }}
```

## Project Structure

Organize W&B projects to mirror your repository structure:

```
Team / Project / Runs
  my-team/
    object-detection/
      run-001-yolov8-baseline
      run-002-yolov8-augmented
      run-003-faster-rcnn-baseline
    segmentation/
      run-001-unet-baseline
```

## Using Pydantic Config with W&B

```python
from pydantic import BaseModel, Field

class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    model_name: str = "yolov8"
    learning_rate: float = Field(gt=0, default=1e-3)
    batch_size: int = Field(ge=1, default=32)
    epochs: int = Field(ge=1, default=100)
    image_size: int = Field(ge=32, default=640)

config = TrainingConfig()

# Pass Pydantic model to W&B as a dict
wandb.init(project="my-cv-project", config=config.model_dump())
```

## Opt-in Pattern

W&B is opt-in: it should only be used when the developer explicitly enables it, and all
code should gracefully handle the case where W&B is not installed or not configured.
Always make W&B optional so the project works without it:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import wandb as wandb_module

def create_logger(use_wandb: bool, project: str, config: dict[str, Any]) -> wandb_module.sdk.wandb_run.Run | None:
    """Create W&B logger if enabled."""
    if not use_wandb:
        return None

    try:
        import wandb
        return wandb.init(project=project, config=config)
    except ImportError:
        print("wandb not installed. Skipping experiment tracking.")
        return None
```

## Integration with PyTorch Lightning

W&B integrates natively with PyTorch Lightning through `WandbLogger`:

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# Create W&B logger
logger = WandbLogger(
    project="my-cv-project",
    name="lightning-experiment-001",
    log_model="all",  # Log all checkpoints as artifacts
    save_dir="logs/",
)

# Define callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val/mAP",
    mode="max",
    save_top_k=3,
    filename="{epoch}-{val_mAP:.3f}",
)

# Create trainer with W&B logger
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    callbacks=[checkpoint_callback],
    accelerator="auto",
)

# Train
trainer.fit(model, datamodule=dm)

# Log additional data after training
logger.experiment.log({"test/mAP": test_map})
```

## Offline Mode

For environments without internet access or during rapid debugging:

```bash
# Set offline mode
export WANDB_MODE=offline

# Later, sync offline runs
wandb sync ./wandb/offline-run-*
```

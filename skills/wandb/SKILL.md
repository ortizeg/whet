---
name: wandb
description: >
  Use this skill when experiment tracking and collaboration run on Weights & Biases —
  logging metrics, managing artifacts, running hyperparameter sweeps, the model registry,
  dataset versioning, and opt-in graceful degradation. Reach for it any time training runs
  need hosted tracking, sweeps, or team sharing and the stack is W&B, even if the user
  just says "track this experiment" or "log the run". For a zero-setup local dashboard see
  tensorboard; for the MLflow alternative see mlflow.
---

# Weights & Biases (W&B) Integration for ML Projects

## Overview

Weights & Biases (W&B or wandb) is an experiment tracking platform purpose-built for machine learning. It provides tools for logging metrics, visualizing training runs, managing datasets and model artifacts, running hyperparameter sweeps, and collaborating with team members. W&B is opt-in: it should only be used when the developer explicitly enables it, and all code should gracefully handle the case where W&B is not installed or not configured.

## Why Use W&B

Experiment tracking is essential for any serious ML project. Without it, developers lose track of which hyperparameters produced which results, cannot reproduce past experiments, and waste time re-running configurations they have already tried. W&B provides:

- **Automatic experiment logging** with zero-friction setup.
- **Interactive dashboards** for comparing runs across metrics.
- **Artifact management** for datasets, models, and predictions.
- **Hyperparameter sweeps** with Bayesian optimization.
- **Team collaboration** with shared workspaces and reports.
- **Model registry** for promoting models through development stages.

## Setup and Configuration

### Installation

```bash
# Using pip
pip install wandb

# Using pixi
pixi add wandb --feature experiment-tracking
```

### Authentication

```bash
# Login with your API key (one-time setup)
wandb login

# Or set the environment variable
export WANDB_API_KEY=your_api_key_here

# For CI/CD environments, use secrets
# GitHub Actions: ${{ secrets.WANDB_API_KEY }}
```

### Project Structure

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

## Experiment Logging

### Basic Initialization

```python
import wandb

# Initialize a new run
run = wandb.init(
    project="my-cv-project",
    name="yolov8-experiment-001",
    config={
        "model": "yolov8",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
        "image_size": 640,
    },
    tags=["baseline", "yolov8", "coco"],
    notes="Baseline YOLOv8 training on COCO subset",
)
```

### Using Pydantic Config with W&B

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

### Opt-in Pattern

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

## Metric Tracking

### Logging Scalars

```python
import wandb

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_map = evaluate(model, val_loader)

    # Log metrics per epoch
    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mAP": val_map,
        "epoch": epoch,
    })
```

### Logging Learning Rate and Gradients

```python
# Log learning rate from scheduler
wandb.log({
    "lr": optimizer.param_groups[0]["lr"],
    "grad_norm": compute_grad_norm(model),
})

# Watch model for automatic gradient and parameter logging
wandb.watch(model, log="all", log_freq=100)
```

### Custom Step Tracking

```python
# Use custom x-axis
wandb.define_metric("train/loss", step_metric="global_step")
wandb.define_metric("val/*", step_metric="epoch")

for step, batch in enumerate(train_loader):
    loss = train_step(model, batch)
    wandb.log({"train/loss": loss, "global_step": step})
```

## Image and Video Logging

### Logging Predictions with Bounding Boxes

```python
import wandb
import numpy as np

def log_predictions(
    images: list[np.ndarray],
    predictions: list[dict],
    class_names: list[str],
    max_images: int = 16,
) -> None:
    """Log prediction images with bounding boxes to W&B."""
    logged_images = []

    for img, pred in zip(images[:max_images], predictions[:max_images]):
        box_data = []
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            box_data.append({
                "position": {
                    "minX": float(box[0]),
                    "minY": float(box[1]),
                    "maxX": float(box[2]),
                    "maxY": float(box[3]),
                },
                "class_id": int(label),
                "box_caption": f"{class_names[label]}: {score:.2f}",
                "scores": {"confidence": float(score)},
            })

        logged_images.append(
            wandb.Image(
                img,
                boxes={"predictions": {
                    "box_data": box_data,
                    "class_labels": {i: name for i, name in enumerate(class_names)},
                }},
            )
        )

    wandb.log({"predictions": logged_images})
```

### Logging Segmentation Masks

```python
import wandb
import numpy as np

def log_segmentation(
    image: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    class_labels: dict[int, str],
) -> None:
    """Log segmentation predictions and ground truth."""
    wandb.log({
        "segmentation": wandb.Image(
            image,
            masks={
                "predictions": {"mask_data": mask_pred, "class_labels": class_labels},
                "ground_truth": {"mask_data": mask_gt, "class_labels": class_labels},
            },
        )
    })
```

### Logging Video

```python
import wandb
import numpy as np

# Log video as a sequence of frames (T, C, H, W)
frames = np.random.randint(0, 255, (30, 3, 480, 640), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=10, format="mp4")})
```

## Artifact Management

Artifacts let you version datasets, models, and other files alongside your runs.

### Saving a Model Artifact

```python
import wandb

def save_model_artifact(
    model_path: str,
    artifact_name: str,
    metadata: dict,
) -> None:
    """Save model as a W&B artifact."""
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata,
        description=f"Model checkpoint with mAP={metadata.get('mAP', 'N/A')}",
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

# Usage
save_model_artifact(
    model_path="checkpoints/best_model.pt",
    artifact_name="yolov8-coco",
    metadata={"mAP": 0.45, "epoch": 50, "image_size": 640},
)
```

### Loading an Artifact

```python
import wandb

def load_model_artifact(artifact_path: str, download_dir: str) -> str:
    """Download a model artifact and return the local path."""
    run = wandb.init(project="my-cv-project")
    artifact = run.use_artifact(artifact_path)
    local_dir = artifact.download(root=download_dir)
    return local_dir

# Usage
model_dir = load_model_artifact(
    artifact_path="my-team/my-cv-project/yolov8-coco:v3",
    download_dir="./downloaded_models",
)
```

### Dataset Artifacts

```python
import wandb

def create_dataset_artifact(
    data_dir: str,
    name: str,
    split: str,
) -> None:
    """Create a dataset artifact from a directory."""
    artifact = wandb.Artifact(
        name=f"{name}-{split}",
        type="dataset",
        metadata={"split": split},
    )
    artifact.add_dir(data_dir)
    wandb.log_artifact(artifact)
```

## Hyperparameter Sweeps

W&B sweeps automate hyperparameter search using Bayesian optimization, grid search, or random search.

### Sweep Configuration

```python
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/mAP", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [8, 16, 32, 64]},
        "optimizer": {"values": ["Adam", "AdamW", "SGD"]},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10,
        "eta": 3,
    },
}

sweep_id = wandb.sweep(sweep_config, project="my-cv-project")
```

### Running a Sweep

```python
import wandb

def train_sweep() -> None:
    """Training function for sweep agent."""
    run = wandb.init()
    config = wandb.config

    model = build_model(config.dropout)
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_map = evaluate(model, val_loader)
        wandb.log({"train/loss": train_loss, "val/mAP": val_map, "epoch": epoch})

# Launch sweep agent
wandb.agent(sweep_id, function=train_sweep, count=50)
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

## Model Registry

The W&B Model Registry provides a central place to manage model versions and promote them through stages:

```python
import wandb

# Link a model artifact to the registry
run = wandb.init(project="my-cv-project")
run.link_artifact(
    artifact=model_artifact,
    target_path="my-team/wandb-registry-model/yolov8-production",
    aliases=["best", "v1.2"],
)
```

## Best Practices

1. **Always use `config` parameter** in `wandb.init()` to log all hyperparameters.
2. **Use structured metric names** with prefixes: `train/loss`, `val/mAP`, `test/precision`.
3. **Log images sparingly** (every N epochs) to avoid slowing down training.
4. **Use tags** to categorize runs: `["baseline", "augmented", "ablation"]`.
5. **Set `WANDB_MODE=offline`** for debugging without uploading data.
6. **Wrap W&B calls** in try/except or behind feature flags for robustness.
7. **Use artifacts** for all datasets and model checkpoints to ensure reproducibility.
8. **Write notes** when starting a run to document the hypothesis being tested.
9. **Create W&B Reports** for sharing results with the team or in papers.
10. **Use `wandb.alert()`** to get notified when training finishes or metrics degrade.

```python
# Alert on training completion
wandb.alert(
    title="Training Complete",
    text=f"Final mAP: {final_map:.4f}",
    level=wandb.AlertLevel.INFO,
)
```

## Offline Mode

For environments without internet access or during rapid debugging:

```bash
# Set offline mode
export WANDB_MODE=offline

# Later, sync offline runs
wandb sync ./wandb/offline-run-*
```

## Summary

W&B is a powerful experiment tracking platform that brings order to the often chaotic process of ML experimentation. By logging metrics, visualizing results, managing artifacts, and enabling hyperparameter optimization, it helps teams iterate faster and make data-driven decisions about model development. Its opt-in nature ensures it can be integrated without imposing a hard dependency on the project.

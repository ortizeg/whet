---
name: tensorboard
description: >
  Use this skill when logging and visualizing PyTorch training locally with TensorBoard —
  scalar metrics, image logging, model-graph visualization, histogram tracking,
  profiling, and zero-setup local experiment monitoring. Reach for it any time you want a
  quick local training dashboard without a hosted service, even if the user just says
  "let me watch the loss" or "log the metrics somewhere". For hosted, collaborative
  tracking with sweeps and a model registry, see wandb or mlflow.
---

# TensorBoard Logging for ML Projects

## Overview

TensorBoard is a visualization toolkit originally built for TensorFlow that has become the standard lightweight logging tool for PyTorch projects as well. It provides real-time visualization of training metrics, model graphs, histograms, images, and profiling data. TensorBoard is opt-in: it should only be used when the developer explicitly enables it, and all code should function without it.

TensorBoard is particularly well-suited as a local, zero-setup visualization tool. Unlike W&B or MLflow, it requires no account, no server, and no internet connection. It reads directly from log files on disk, making it the fastest path from training to visualization.

## Why Use TensorBoard

- **Zero infrastructure** required. No server, no account, no API key.
- **Real-time visualization** during training.
- **Native PyTorch support** through `torch.utils.tensorboard`.
- **Lightweight** with minimal overhead on training performance.
- **Rich visualizations** including scalars, images, histograms, and computation graphs.
- **Profiling support** for identifying training bottlenecks.
- **Built into PyTorch Lightning** with no additional configuration needed.

## Setup and Configuration

### Installation

```bash
# Using pip
pip install tensorboard

# Using pixi
pixi add tensorboard --feature experiment-tracking

# TensorBoard is included with PyTorch Lightning
pip install pytorch-lightning  # includes tensorboard
```

### Launching TensorBoard

```bash
# Basic launch
tensorboard --logdir=logs/

# Specify port
tensorboard --logdir=logs/ --port 6006

# Compare multiple experiments
tensorboard --logdir=logs/exp1:experiment_1,logs/exp2:experiment_2

# Bind to all interfaces (for remote access)
tensorboard --logdir=logs/ --host 0.0.0.0
```

### Using with VS Code

VS Code has built-in TensorBoard support:

1. Open the Command Palette (Cmd+Shift+P).
2. Search for "Python: Launch TensorBoard".
3. Select the log directory.
4. TensorBoard opens in a VS Code tab.

## Scalar Logging

Scalars are the most common type of data logged to TensorBoard. They track metrics over time.

### Basic Scalar Logging

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter(log_dir="logs/experiment_001")

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_map = evaluate(model, val_loader)

    # Log scalars
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Metrics/mAP", val_map, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

# Always close the writer
writer.close()
```

### Logging Multiple Scalars Together

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log multiple scalars on the same plot
writer.add_scalars("Loss", {
    "train": train_loss,
    "val": val_loss,
}, epoch)

writer.add_scalars("Per_Class_AP", {
    "car": 0.52,
    "person": 0.48,
    "bike": 0.35,
}, epoch)
```

### Opt-in Pattern

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """Optional TensorBoard logger."""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._writer: SummaryWriter | None = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("tensorboard not installed. Skipping logging.")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        """Log an image."""
        if self._writer is not None:
            self._writer.add_image(tag, image, step)

    def close(self) -> None:
        """Close the writer."""
        if self._writer is not None:
            self._writer.close()
```

## Image Logging

TensorBoard can display images, which is invaluable for computer vision projects.

### Logging Individual Images

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log a single image (C, H, W) format, values in [0, 1]
writer.add_image("sample/input", image_tensor, epoch)

# Log prediction vs ground truth
writer.add_image("sample/prediction", pred_image, epoch)
writer.add_image("sample/ground_truth", gt_image, epoch)
```

### Logging Image Grids

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter(log_dir="logs/experiment_001")

# Create a grid of images (N, C, H, W)
images = torch.stack([batch[i] for i in range(min(16, len(batch)))])
grid = make_grid(images, nrow=4, normalize=True, padding=2)
writer.add_image("batch/inputs", grid, epoch)

# Log augmented vs original
original_grid = make_grid(original_images[:8], nrow=4, normalize=True)
augmented_grid = make_grid(augmented_images[:8], nrow=4, normalize=True)
writer.add_image("augmentation/original", original_grid, epoch)
writer.add_image("augmentation/augmented", augmented_grid, epoch)
```

### Logging Images with Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

def log_detection_figure(
    writer: SummaryWriter,
    image: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    step: int,
) -> None:
    """Log detection results as a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color="red", fontsize=10)
    ax.axis("off")
    writer.add_figure("detections", fig, step)
    plt.close(fig)
```

## Histogram Logging

Histograms show the distribution of values over time, useful for monitoring weights and gradients.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log weight and gradient distributions
for name, param in model.named_parameters():
    writer.add_histogram(f"weights/{name}", param.data, epoch)
    if param.grad is not None:
        writer.add_histogram(f"gradients/{name}", param.grad, epoch)

# Log activation distributions
def hook_fn(module, input, output, name, writer, step):
    writer.add_histogram(f"activations/{name}", output.detach(), step)

# Register hooks
for name, module in model.named_modules():
    module.register_forward_hook(
        lambda m, i, o, n=name: hook_fn(m, i, o, n, writer, global_step)
    )
```

## Graph Visualization

TensorBoard can visualize the computation graph of your model.

```python
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log model graph
dummy_input = torch.randn(1, 3, 640, 640)
writer.add_graph(model, dummy_input)
writer.close()
```

## Hyperparameter Tuning

TensorBoard's HParams plugin allows comparing runs across hyperparameter configurations.

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

writer = SummaryWriter(log_dir="logs/hparams/run_001")

# Define hyperparameters and metrics
hparam_dict = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "AdamW",
    "dropout": 0.1,
}
metric_dict = {
    "hparam/mAP": best_map,
    "hparam/val_loss": best_val_loss,
}

# Log hyperparameters with their resulting metrics
writer.add_hparams(hparam_dict, metric_dict)
writer.close()
```

### Running Multiple Hyperparameter Experiments

```python
from torch.utils.tensorboard import SummaryWriter

learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        run_name = f"lr_{lr}_bs_{bs}"
        writer = SummaryWriter(log_dir=f"logs/hparams/{run_name}")

        # Train with these hyperparameters
        best_map = train_and_evaluate(lr=lr, batch_size=bs)

        writer.add_hparams(
            {"learning_rate": lr, "batch_size": bs},
            {"hparam/mAP": best_map},
        )
        writer.close()
```

## Integration with PyTorch Lightning

PyTorch Lightning has built-in TensorBoard support as the default logger:

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# Create TensorBoard logger
logger = TensorBoardLogger(
    save_dir="logs/",
    name="object-detection",
    version="experiment_001",
    default_hp_metric=False,
)

# Create trainer (TensorBoard is the default logger)
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="auto",
)

# In your LightningModule, logging is automatic
class DetectionModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.evaluate(batch)
        self.log("val/loss", loss)
        self.log("val/mAP", metrics["mAP"])

    def on_validation_epoch_end(self):
        # Log images at end of validation
        if self.current_epoch % 5 == 0:
            tensorboard = self.logger.experiment
            tensorboard.add_image("val/predictions", pred_grid, self.current_epoch)
```

## Custom Summary Writing

For advanced use cases, write custom summaries:

```python
from torch.utils.tensorboard import SummaryWriter
import json

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log text
writer.add_text("config", json.dumps(config_dict, indent=2), 0)
writer.add_text("notes", "Baseline experiment with default augmentation", 0)

# Log embeddings (useful for feature visualization)
features = model.extract_features(images)  # (N, D)
metadata = [class_names[label] for label in labels]
writer.add_embedding(
    features,
    metadata=metadata,
    label_img=images,
    global_step=epoch,
    tag="feature_embeddings",
)

# Log precision-recall curve
writer.add_pr_curve("PR/car", labels_car, predictions_car, epoch)
```

## Profiling

TensorBoard's profiling plugin helps identify performance bottlenecks:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# Profile training
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("logs/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 1:
            break
        with record_function("train_step"):
            loss = train_step(model, batch, optimizer)
        prof.step()

# View in TensorBoard under the "PyTorch Profiler" tab
```

## Remote Access

To access TensorBoard running on a remote machine:

```bash
# On the remote machine
tensorboard --logdir=logs/ --host 0.0.0.0 --port 6006

# On your local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-host

# Then open http://localhost:6006 in your browser
```

### Using TensorBoard.dev (Cloud)

```bash
# Upload logs to TensorBoard.dev (public sharing)
tensorboard dev upload --logdir logs/ --name "My Experiment" --description "Baseline results"

# List uploaded experiments
tensorboard dev list

# Delete an experiment
tensorboard dev delete --experiment_id EXPERIMENT_ID
```

## Best Practices

1. **Use consistent tag naming**: `{split}/{metric}` (e.g., `train/loss`, `val/mAP`).
2. **Log images periodically** (every 5-10 epochs) to avoid large log files.
3. **Use `flush()` or `close()`** to ensure data is written to disk.
4. **Organize log directories** by experiment: `logs/{experiment_name}/{run_name}`.
5. **Log hyperparameters** with `add_hparams` for comparison across runs.
6. **Profile early** to catch data loading bottlenecks before long training runs.
7. **Use `add_scalars`** (plural) to overlay related metrics on the same chart.
8. **Add `.gitignore` entries** for log directories: `logs/`, `runs/`.
9. **Wrap TensorBoard** behind an opt-in flag for flexibility.
10. **Combine with Lightning** for automatic metric logging with minimal code.

```python
# Add to .gitignore
# logs/
# runs/
# lightning_logs/
```

## Summary

TensorBoard is the simplest, most lightweight experiment visualization tool available for PyTorch projects. It requires no accounts, no servers, and no internet connection. Its deep integration with PyTorch (through `SummaryWriter`) and PyTorch Lightning (as the default logger) makes it the go-to choice for local development and quick experiments. For team collaboration or advanced features like artifact management and sweeps, consider supplementing it with W&B or MLflow.

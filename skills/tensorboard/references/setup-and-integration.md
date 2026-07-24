# TensorBoard Setup, Opt-in Wiring, and Remote Access

Scope: installing TensorBoard, launching and comparing runs, making logging optional, integrating with PyTorch Lightning, and viewing logs from a remote machine.

## Contents

- [Installation](#installation)
- [Launching TensorBoard](#launching-tensorboard)
- [Using with VS Code](#using-with-vs-code)
- [Opt-in Pattern](#opt-in-pattern)
- [Integration with PyTorch Lightning](#integration-with-pytorch-lightning)
- [Remote Access](#remote-access)
- [Using TensorBoard.dev (Cloud)](#using-tensorboarddev-cloud)
- [Gitignore Entries](#gitignore-entries)

## Installation

```bash
# Using pip
pip install tensorboard

# Using pixi
pixi add tensorboard --feature experiment-tracking

# TensorBoard is included with PyTorch Lightning
pip install pytorch-lightning  # includes tensorboard
```

## Launching TensorBoard

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

## Using with VS Code

VS Code has built-in TensorBoard support:

1. Open the Command Palette (Cmd+Shift+P).
2. Search for "Python: Launch TensorBoard".
3. Select the log directory.
4. TensorBoard opens in a VS Code tab.

## Opt-in Pattern

TensorBoard is opt-in: it should only be used when the developer explicitly enables it, and
all code should function without it.

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

## Remote Access

To access TensorBoard running on a remote machine:

```bash
# On the remote machine
tensorboard --logdir=logs/ --host 0.0.0.0 --port 6006

# On your local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@remote-host

# Then open http://localhost:6006 in your browser
```

## Using TensorBoard.dev (Cloud)

```bash
# Upload logs to TensorBoard.dev (public sharing)
tensorboard dev upload --logdir logs/ --name "My Experiment" --description "Baseline results"

# List uploaded experiments
tensorboard dev list

# Delete an experiment
tensorboard dev delete --experiment_id EXPERIMENT_ID
```

## Gitignore Entries

```python
# Add to .gitignore
# logs/
# runs/
# lightning_logs/
```

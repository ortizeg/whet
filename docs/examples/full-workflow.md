# Full Workflow Example

End-to-end walkthrough of creating an object detection project, from initialization through training, export, and CI/CD setup.

## Step 1: Create the Project

```
claude

You: "Create a pytorch-training-project called 'yolo-detector' with wandb tracking"
```

Claude generates the full project structure using the Master Skill, including W&B integration.

## Step 2: Define Configuration Models

The Expert Coder creates Pydantic configs following the `pydantic` skill:

```python
# src/yolo_detector/configs.py
from __future__ import annotations

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Model architecture configuration."""
    backbone: str = Field(default="resnet50", description="Backbone network")
    num_classes: int = Field(ge=1, description="Number of object classes")
    input_size: int = Field(default=640, ge=32, description="Input image size")
    dropout: float = Field(ge=0, le=1, default=0.1)

    model_config = {"frozen": True}

class DataConfig(BaseModel):
    """Dataset configuration."""
    data_dir: str = Field(description="Path to COCO dataset")
    batch_size: int = Field(ge=1, default=16)
    num_workers: int = Field(ge=0, default=4)
    image_size: int = Field(default=640, ge=32)

class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    learning_rate: float = Field(gt=0, default=1e-3)
    weight_decay: float = Field(ge=0, default=0.01)
    max_epochs: int = Field(ge=1, default=100)
    gradient_clip_val: float = Field(ge=0, default=1.0)
```

## Step 3: Implement the LightningModule

Following the `pytorch-lightning` skill:

```python
# src/yolo_detector/models/detector.py
from __future__ import annotations

import lightning as L
import torch
from torch import nn

from yolo_detector.configs import ModelConfig

class ObjectDetector(L.LightningModule):
    """Object detection model with Pydantic config."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.backbone = self._build_backbone()
        self.head = self._build_detection_head()

    def _build_backbone(self) -> nn.Module:
        """Build backbone network."""
        import torchvision.models as models
        backbone = getattr(models, self.config.backbone)(pretrained=True)
        return nn.Sequential(*list(backbone.children())[:-2])

    def _build_detection_head(self) -> nn.Module:
        """Build detection head."""
        return nn.Sequential(
            nn.Conv2d(2048, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, self.config.num_classes + 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.head(features)

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Training step with loss logging."""
        images, targets = batch
        predictions = self(images)
        loss = self._compute_loss(predictions, targets)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> None:
        """Validation step with mAP logging."""
        images, targets = batch
        predictions = self(images)
        loss = self._compute_loss(predictions, targets)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure AdamW with cosine annealing."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate if hasattr(self.config, 'learning_rate') else 1e-3,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100
        )
        return [optimizer], [scheduler]
```

## Step 4: Implement the DataModule

```python
# src/yolo_detector/data/datamodule.py
from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from yolo_detector.configs import DataConfig

class COCODataModule(L.LightningDataModule):
    """COCO dataset DataModule with Pydantic config."""

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.config = config

    def setup(self, stage: str) -> None:
        """Setup train/val datasets."""
        if stage == "fit":
            self.train_dataset = self._create_dataset("train")
            self.val_dataset = self._create_dataset("val")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
```

## Step 5: Create Hydra Configuration

Following the `hydra-config` skill:

```yaml
# configs/config.yaml
defaults:
  - model: resnet50
  - data: coco
  - _self_

trainer:
  max_epochs: 100
  accelerator: auto
  precision: 16-mixed

# configs/model/resnet50.yaml
backbone: resnet50
num_classes: 80
input_size: 640
dropout: 0.1
```

## Step 6: Run Training

```bash
# Default training
uv run python -m yolo_detector.train

# Override config
uv run python -m yolo_detector.train model.backbone=efficientnet_b0 trainer.max_epochs=50

# Debug mode
uv run python -m yolo_detector.train trainer.fast_dev_run=true
```

## Step 7: Export to ONNX

Following the `onnx` skill:

```python
# scripts/export_onnx.py
import torch
from yolo_detector.models.detector import ObjectDetector
from yolo_detector.configs import ModelConfig

config = ModelConfig(backbone="resnet50", num_classes=80)
model = ObjectDetector.load_from_checkpoint("checkpoints/best.ckpt", config=config)
model.eval()

dummy = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model, dummy, "models/detector.onnx",
    opset_version=17,
    input_names=["image"],
    output_names=["detections"],
    dynamic_axes={"image": {0: "batch"}, "detections": {0: "batch"}},
)
```

## Step 8: CI/CD Setup

The Code Review and Test Engineer agents are already configured:

```yaml
# .github/workflows/code-review.yml — checks formatting, linting, types
# .github/workflows/test.yml — runs tests, enforces 80% coverage
```

Every PR must pass both before merge.

## How Skills Guide Each Decision

| Decision | Skill Used |
|----------|-----------|
| Config validation | `pydantic` |
| Model structure | `pytorch-lightning` |
| Type annotations | `code-quality` |
| Video/image I/O | `abstraction-patterns` |
| Experiment tracking | `wandb` |
| Model export | `onnx` |
| CI enforcement | `github-actions` |
| Environment setup | `pixi` |

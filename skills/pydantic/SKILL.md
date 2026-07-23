---
name: pydantic
description: >
  Use this skill when validating configuration or data structures with Pydantic V2 —
  frozen/immutable config models, discriminated unions, custom validators, settings
  management, and strict load-time validation. Reach for it any time you'd otherwise pass
  around raw dicts or plain dataclasses for config or structured data, even if the user
  doesn't say "Pydantic". This is for plain data and config validation; for validating an
  LLM's output and building agents see pydantic-ai, and for Hydra config composition see
  hydra-config.
---

# Pydantic Skill

Configuration and data validation with Pydantic V2 and strict typing. Validate all config at load time, never inside training loops. Every config, data structure, and API payload uses Pydantic.

## Level 1: Immutable Configuration (Frozen Models)

Use frozen models for training/inference configs that should never be modified after creation. This is the default for all configuration objects.

```python
"""Training configuration with strict Pydantic validation."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    model_config = {"frozen": True, "extra": "forbid"}

    data_dir: Path
    batch_size: int = Field(gt=0, le=4096, default=32)
    num_workers: int = Field(ge=0, le=32, default=4)
    image_size: tuple[int, int] = (224, 224)
    pin_memory: bool = True

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir(cls, v: Path) -> Path:
        """Ensure data directory exists."""
        if not v.exists():
            msg = f"Data directory does not exist: {v}"
            raise ValueError(msg)
        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        """Ensure image dimensions are positive and reasonable."""
        h, w = v
        if h <= 0 or w <= 0:
            msg = f"Image dimensions must be positive, got ({h}, {w})"
            raise ValueError(msg)
        if h > 4096 or w > 4096:
            msg = f"Image dimensions too large: ({h}, {w}), max is 4096"
            raise ValueError(msg)
        return v


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    model_config = {"frozen": True, "extra": "forbid"}

    backbone: str = "resnet50"
    num_classes: int = Field(gt=0)
    pretrained: bool = True
    dropout: float = Field(ge=0.0, le=1.0, default=0.0)


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = "adamw"
    learning_rate: float = Field(gt=0.0, le=1.0, default=1e-3, alias="lr")
    weight_decay: float = Field(ge=0.0, default=1e-2)
    momentum: float = Field(ge=0.0, le=1.0, default=0.9)

    @field_validator("name")
    @classmethod
    def validate_optimizer_name(cls, v: str) -> str:
        """Ensure optimizer name is supported."""
        valid_names = {"adam", "adamw", "sgd", "rmsprop"}
        if v.lower() not in valid_names:
            msg = f"Unsupported optimizer: {v}. Choose from {valid_names}"
            raise ValueError(msg)
        return v.lower()


class TrainConfig(BaseModel):
    """Top-level training configuration.

    Composes all sub-configs into a single validated object.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    max_epochs: int = Field(gt=0, le=10000, default=100)
    seed: int = 42
    experiment_name: str = "default"
```

### Level 1 Rules

1. **Always set `frozen = True`** -- configs must be immutable after creation.
2. **Always set `extra = "forbid"`** -- reject unknown fields to catch typos.
3. **Use `Field()` with constraints** -- `gt`, `ge`, `lt`, `le` for numeric bounds.
4. **Use `field_validator`** for complex validation that Field constraints cannot express.
5. **Compose configs hierarchically** -- one top-level config that contains sub-configs.
6. **Document every field** with a docstring in the class.

## Level 2: Mutable Data Structures

Use non-frozen models for data structures that need to be updated during processing, such as tracking state, results, or intermediate computations.

```python
"""Mutable data structures for tracking training progress."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TrainingMetrics(BaseModel):
    """Mutable metrics tracked during training."""

    model_config = {"extra": "forbid"}

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = float("inf")
    best_val_loss: float = float("inf")
    learning_rate: float = 0.0
    samples_processed: int = 0

    def update_epoch(self, train_loss: float, val_loss: float, lr: float) -> None:
        """Update metrics after completing an epoch."""
        self.epoch += 1
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.learning_rate = lr
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss


class DetectionResult(BaseModel):
    """Single object detection result. bbox is (x1, y1, x2, y2)."""

    model_config = {"extra": "forbid"}

    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Ensure bounding box coordinates are valid."""
        x1, y1, x2, y2 = v
        if x2 <= x1 or y2 <= y1:
            msg = f"Invalid bbox: x2 must be > x1 and y2 must be > y1, got ({x1}, {y1}, {x2}, {y2})"
            raise ValueError(msg)
        return v


class FrameDetections(BaseModel):
    """All detections in a single frame."""

    model_config = {"extra": "forbid"}

    frame_id: int = Field(ge=0)
    timestamp: float = Field(ge=0.0)
    detections: list[DetectionResult] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)

    @property
    def num_detections(self) -> int:
        """Return total number of detections."""
        return len(self.detections)

    def filter_by_confidence(self, threshold: float) -> list[DetectionResult]:
        """Return detections above the confidence threshold."""
        return [d for d in self.detections if d.confidence >= threshold]
```

### Level 2 Rules

1. **Omit `frozen = True`** -- these models need to be mutable.
2. **Still set `extra = "forbid"`** -- even mutable models should reject unknown fields.
3. **Add methods for common operations** -- like `update_epoch()` and `filter_by_confidence()`.
4. **Use properties for derived values** -- like `num_detections`.

## Level 2.5: Selective Strict Validation

Use `model_validator` with mode `"before"` for cases where you need to transform input data before standard validation. This is useful for loading configs from multiple formats.

```python
"""Selective validation for flexible config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class AugmentationConfig(BaseModel):
    """Augmentation config. Loads from a dict or a path to a YAML file."""

    model_config = {"frozen": True, "extra": "forbid"}

    horizontal_flip_prob: float = Field(ge=0.0, le=1.0, default=0.5)
    vertical_flip_prob: float = Field(ge=0.0, le=1.0, default=0.0)
    rotation_limit: int = Field(ge=0, le=180, default=15)
    brightness_limit: float = Field(ge=0.0, le=1.0, default=0.2)
    contrast_limit: float = Field(ge=0.0, le=1.0, default=0.2)

    @model_validator(mode="before")
    @classmethod
    def load_from_file_if_string(cls, data: Any) -> Any:
        """If data is a string path, load YAML from that file."""
        if isinstance(data, str):
            import yaml

            path = Path(data)
            if not path.exists():
                msg = f"Config file not found: {path}"
                raise ValueError(msg)
            with open(path) as f:
                return yaml.safe_load(f)
        return data


class ExperimentConfig(BaseModel):
    """Experiment configuration with automatic defaults based on task."""

    model_config = {"frozen": True, "extra": "forbid"}

    task: str
    backbone: str = "resnet50"
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def set_defaults_by_task(self) -> Self:
        """Set sensible defaults based on the task type."""
        # Use object.__setattr__ because the model is frozen
        if self.task == "segmentation" and self.image_size == (224, 224):
            object.__setattr__(self, "image_size", (512, 512))
        if self.task == "detection" and self.backbone == "resnet50":
            object.__setattr__(self, "backbone", "resnet50-fpn")
        return self
```

### Level 2.5 Rules

1. **Use `mode="before"`** for input transformation (loading from files, normalizing formats).
2. **Use `mode="after"`** for cross-field validation and conditional defaults.
3. **Use `object.__setattr__`** to modify frozen models inside `model_validator(mode="after")`.
4. **Always return `self`** from `mode="after"` validators (return type is `Self`).

## Nested Configuration Pattern

For complex projects, compose configs from multiple files:

```python
"""Composable nested configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    model_config = {"frozen": True, "extra": "forbid"}

    name: str = "cosine"
    warmup_epochs: int = Field(ge=0, default=5)
    min_lr: float = Field(ge=0.0, default=1e-6)


class FullConfig(BaseModel):
    """Full experiment configuration composed from nested configs."""

    model_config = {"frozen": True, "extra": "forbid"}

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    max_epochs: int = Field(gt=0, default=100)
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> FullConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated FullConfig instance.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValidationError: If the config file contains invalid values.
        """
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(raw)

    @classmethod
    def from_yaml_with_overrides(
        cls,
        path: str | Path,
        overrides: dict[str, Any],
    ) -> FullConfig:
        """Load config from YAML and apply overrides.

        Args:
            path: Path to the base YAML configuration file.
            overrides: Dictionary of dot-separated key overrides.

        Returns:
            Validated FullConfig with overrides applied.
        """
        path = Path(path)
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        # Apply nested overrides using dot notation
        for key, value in overrides.items():
            parts = key.split(".")
            target = raw
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value

        return cls.model_validate(raw)
```

### Usage Example

```python
# Load from YAML, optionally with dot-notation overrides
config = FullConfig.from_yaml("configs/experiment/baseline.yaml")
config = FullConfig.from_yaml_with_overrides(
    "configs/experiment/baseline.yaml",
    overrides={"optimizer.lr": 5e-4, "data.batch_size": 64, "max_epochs": 200},
)

# Access nested values with full type safety
print(config.optimizer.learning_rate)  # float
print(config.data.image_size)          # tuple[int, int]
```

The matching YAML has top-level `data`, `model`, `optimizer`, `scheduler` keys
mirroring the sub-config field names (use `lr` alias under `optimizer`).

## Testing Pydantic Configs

```python
"""Tests for configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from my_project.config import DataConfig, ModelConfig, TrainConfig


def test_valid_data_config(tmp_path) -> None:
    """Test creating a valid data config."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = DataConfig(data_dir=data_dir, batch_size=64)
    assert config.batch_size == 64
    assert config.num_workers == 4  # default


def test_invalid_batch_size() -> None:
    """Test that invalid batch size raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than 0"):
        DataConfig(data_dir="/tmp", batch_size=0)


def test_extra_fields_rejected() -> None:
    """Test that unknown fields are rejected."""
    with pytest.raises(ValidationError, match="Extra inputs"):
        ModelConfig(num_classes=10, unknown_field="value")


def test_frozen_model_immutable(tmp_path) -> None:
    """Test that frozen configs cannot be modified."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = DataConfig(data_dir=data_dir)
    with pytest.raises(ValidationError):
        config.batch_size = 128  # type: ignore[misc]


def test_config_from_yaml(tmp_path) -> None:
    """Test loading full config from YAML."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
    data:
      data_dir: /tmp/data
      batch_size: 32
    model:
      num_classes: 10
    optimizer:
      name: adamw
      lr: 0.001
    """)
    config = TrainConfig.from_yaml(config_file)
    assert config.model.num_classes == 10
```

## Anti-Patterns to Avoid

1. **Never use plain dicts for configuration** -- always use Pydantic models.
2. **Never use `extra = "allow"`** -- unknown fields hide bugs and typos.
3. **Never skip validation** -- always call `model_validate()` or construct via the constructor.
4. **Never use mutable defaults** -- use `Field(default_factory=list)` instead of `default=[]`.
5. **Never store secrets in config models** -- use environment variables with `SecretStr`.
6. **Never ignore ValidationError** -- fix the config, do not catch and suppress.

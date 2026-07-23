# Pydantic

The Pydantic skill enforces rigorous data validation patterns using Pydantic v2, ensuring type safety and validation at runtime boundaries in CV/ML projects.

**Skill directory:** `skills/pydantic/`

## Purpose

ML projects are prone to silent data errors -- wrong image dimensions, misconfigured hyperparameters, invalid file paths. This skill teaches Claude Code to use Pydantic v2 in strict mode for all configuration, data interchange, and API contracts. Strict mode rejects type coercion, so a float passed where an int is expected raises an error immediately rather than silently truncating.

## When to Use

- Defining configuration schemas for training, inference, or data processing
- Building API request/response models for inference services
- Validating dataset metadata and annotation formats
- Any boundary where untrusted data enters your system

## Key Patterns

### Strict Configuration Model

```python
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

class TrainingConfig(BaseModel, strict=True):
    """Configuration for model training."""

    model_name: str = Field(description="Name of the model architecture")
    num_classes: int = Field(gt=0, description="Number of output classes")
    learning_rate: float = Field(gt=0.0, lt=1.0, default=1e-3)
    batch_size: int = Field(gt=0, default=32)
    max_epochs: int = Field(gt=0, default=100)
    data_dir: Path = Field(description="Root directory for training data")
    precision: str = Field(default="16-mixed", pattern=r"^(16-mixed|32|bf16-mixed)$")

    @field_validator("data_dir")
    @classmethod
    def validate_data_dir_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Data directory does not exist: {v}")
        return v
```

### Nested Configuration

```python
class ModelConfig(BaseModel, strict=True):
    backbone: str = Field(default="resnet50")
    pretrained: bool = Field(default=True)
    dropout: float = Field(ge=0.0, le=1.0, default=0.1)

class DataConfig(BaseModel, strict=True):
    root: Path
    image_size: tuple[int, int] = Field(default=(224, 224))
    augment: bool = Field(default=True)

class ExperimentConfig(BaseModel, strict=True):
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
```

### API Models for Inference

```python
class PredictionRequest(BaseModel, strict=True):
    image_b64: str = Field(description="Base64-encoded image")
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    max_detections: int = Field(gt=0, default=100)

class BoundingBox(BaseModel, strict=True):
    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(ge=0.0)
    y_max: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: str

class PredictionResponse(BaseModel, strict=True):
    detections: list[BoundingBox]
    inference_time_ms: float
```

## Anti-Patterns to Avoid

- Never use `dict` when you could use a Pydantic model -- lose validation and documentation
- Never use `model_config = ConfigDict(strict=False)` -- defeats the purpose of this skill
- Avoid `Any` type in model fields -- be explicit about expected types
- Do not use `.dict()` (Pydantic v1) -- use `.model_dump()` (Pydantic v2)

## Combines Well With

- **Hydra Config** -- Pydantic models as structured config targets
- **Master Skill** -- Type annotation conventions
- **Testing** -- Parameterized validation tests
- **PyPI** -- Well-typed public API surfaces

## Full Reference

See [`skills/pydantic/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/pydantic/SKILL.md) for patterns covering custom validators, serialization, and Pydantic settings for environment variable loading.

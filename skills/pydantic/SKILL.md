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

## Essential Core: Frozen Configuration Models

Frozen, `extra="forbid"` models composed hierarchically into one top-level config is the default
shape for every configuration object. Bounds go in `Field()`; anything a constraint cannot
express goes in a `field_validator`.

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

## Conventions

1. **Always set `frozen = True`** -- configs must be immutable after creation.
2. **Always set `extra = "forbid"`** -- reject unknown fields to catch typos.
3. **Use `Field()` with constraints** -- `gt`, `ge`, `lt`, `le` for numeric bounds.
4. **Use `field_validator`** for complex validation that Field constraints cannot express.
5. **Compose configs hierarchically** -- one top-level config that contains sub-configs.
6. **Document every field** with a docstring in the class.
7. **Raise `ValueError` inside validators** (assigned to `msg` first) so Pydantic wraps it into a
   `ValidationError` with field location.
8. **Validate once at load time** -- pass the validated object down, never re-parse in a loop.

## Anti-Patterns to Avoid

1. **Never use plain dicts for configuration** -- always use Pydantic models.
2. **Never use `extra = "allow"`** -- unknown fields hide bugs and typos.
3. **Never skip validation** -- always call `model_validate()` or construct via the constructor.
4. **Never use mutable defaults** -- use `Field(default_factory=list)` instead of `default=[]`.
5. **Never store secrets in config models** -- use environment variables with `SecretStr`.
6. **Never ignore ValidationError** -- fix the config, do not catch and suppress.
7. **Never use `object.__setattr__` outside a `model_validator(mode="after")`** -- it defeats the
   point of freezing the model.

## Deep dives

- `references/mutable-data-models.md` — read when modeling state that changes during processing: training metrics, detection results, per-frame aggregates.
- `references/validators-and-transforms.md` — read when you need `model_validator` for cross-field rules, conditional defaults, or reshaping input before validation.
- `references/nested-config-composition.md` — read when composing a full experiment config, loading it from YAML, or applying dot-notation overrides.
- `references/testing-configs.md` — read when writing tests that prove the constraints, frozen-ness, and YAML loading actually work.

# Validators and Input Transforms

Using `model_validator` to reshape input before validation and to apply cross-field logic afterwards — including how to mutate a frozen model from an `after` validator.

## Contents

- [Before and After Validators](#before-and-after-validators)
- [Rules](#rules)
- [Choosing Between field_validator and model_validator](#choosing-between-field_validator-and-model_validator)

## Before and After Validators

Use `model_validator` with mode `"before"` for cases where you need to transform input data
before standard validation. This is useful for loading configs from multiple formats.

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

## Rules

1. **Use `mode="before"`** for input transformation (loading from files, normalizing formats).
2. **Use `mode="after"`** for cross-field validation and conditional defaults.
3. **Use `object.__setattr__`** to modify frozen models inside `model_validator(mode="after")`.
4. **Always return `self`** from `mode="after"` validators (return type is `Self`).

## Notes

- **`mode="before"` receives raw input of any type** — a dict, a string, whatever was passed —
  which is exactly why it can accept a file path where a dict is expected. It must be a
  `@classmethod` and must return something the normal validation pass can consume.
- **`mode="after"` receives the fully validated model**, so every field is present and typed.
  This is where invariants spanning two fields belong ("`min_lr` must be below `learning_rate`",
  "segmentation implies a larger `image_size`").
- **`object.__setattr__` bypasses the frozen guard.** It is the sanctioned escape hatch inside
  an `after` validator; using it anywhere else defeats the point of freezing the model.
- **Conditional defaults are only detectable by comparing against the declared default**
  (`self.image_size == (224, 224)`). This cannot distinguish "user explicitly asked for 224" from
  "user said nothing". When that distinction matters, declare the field as `| None = None` and
  fill it in the `after` validator instead.
- **Raise `ValueError`, not custom exception types**, inside validators. Pydantic wraps
  `ValueError`/`AssertionError` into a `ValidationError` with field location information;
  other exceptions propagate raw and lose that context.
- Assign the message to a variable before raising (`msg = ...; raise ValueError(msg)`) to satisfy
  the project's Ruff configuration.

## Choosing Between field_validator and model_validator

- **`Field(...)` constraints** — bounds, lengths, regex. Always prefer these; they are declarative
  and show up in the JSON schema.
- **`field_validator`** — single-field logic that constraints cannot express: "this directory
  must exist", "this optimizer name must be in the supported set", "x2 must exceed x1 within one
  tuple". Runs per field, sees only that field's value.
- **`model_validator`** — anything needing more than one field, or needing to reshape the input
  before field validation happens.

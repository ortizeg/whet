# Nested Configuration Composition

Composing a full experiment config from sub-configs, loading it from YAML, and applying dot-notation overrides.

## Contents

- [Composable Config](#composable-config)
- [Usage](#usage)
- [Notes](#notes)

## Composable Config

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

## Usage

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

## Notes

- **`model_validate(raw)` is the single validation gate.** Overrides are applied to the raw dict
  *before* validation, so an invalid override (`{"data.batch_size": 0}`) fails with the same
  `ValidationError` as an invalid file — there is no path that produces a half-validated object.
- **`setdefault(part, {})` walks and creates** intermediate dicts, so an override can target a
  section the YAML omitted entirely (e.g. `scheduler.warmup_epochs` when the file has no
  `scheduler` key at all).
- **Dot-notation keys use the field's alias where one exists** — `optimizer.lr`, not
  `optimizer.learning_rate`, because `OptimizerConfig` declares `alias="lr"`.
- **`yaml.safe_load`, never `yaml.load`.** `safe_load` cannot instantiate arbitrary Python
  objects from the file.
- **Sub-configs with sensible defaults use `Field(default_factory=SchedulerConfig)`** so the
  whole section can be omitted from YAML. Sub-configs without defaults (`data`, `model`,
  `optimizer`) are required, which is what forces every experiment to state them explicitly.
- **`from_yaml` raises `FileNotFoundError` explicitly** rather than letting `open()` fail deeper
  in the call stack — the error names the config path the user actually passed.
- Loading happens once at process start. Never re-read config inside a training loop; pass the
  validated object down.

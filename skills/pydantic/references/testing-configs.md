# Testing Pydantic Configs

What to assert about a config model: that valid input works, and that every guard actually rejects bad input.

## Test Suite

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

## What to Cover

- **One happy-path test per config** asserting both an explicit value and a default. Defaults are
  part of the contract and silently changing one breaks callers.
- **One rejection test per guard.** Every `Field` constraint, `field_validator`, and
  `model_validator` deserves a test that proves it fires — an untested validator is often a
  validator with an inverted condition.
- **`extra = "forbid"`** — assert that an unknown field raises. This is the guard that catches
  typos in YAML, and it is worth a test in its own right.
- **`frozen = True`** — assert that assignment raises. Note that Pydantic V2 raises
  `ValidationError` (not `TypeError`) on frozen-field assignment.
- **YAML round-trip** — write a real file to `tmp_path` and load it. This exercises aliases
  (`lr` → `learning_rate`), nested composition, and the loader together.
- **Conditional defaults from `model_validator(mode="after")`** — assert the transformed value,
  e.g. that `task="segmentation"` yields `image_size == (512, 512)`.

## Techniques

- **`match=` on `pytest.raises`** pins the failure to the specific constraint. `match="greater
  than 0"` proves the `gt=0` bound fired rather than some unrelated error swallowing the test.
- **Use `tmp_path`** for anything a validator checks on disk (`data_dir` existence). Never point
  tests at a real dataset directory.
- **`# type: ignore[misc]`** is required on the frozen-assignment line because MyPy correctly
  flags assignment to a frozen field — the ignore documents that the failure is the point.
- Config tests are fast and pure; run them in the default CI test job, not behind a GPU marker.

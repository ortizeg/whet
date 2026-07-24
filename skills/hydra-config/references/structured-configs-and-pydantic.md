# Structured Configs and Pydantic Validation

Scope: defining Hydra dataclass configs, registering them with the ConfigStore, and converting the composed config into Pydantic models for runtime validation.

## Contents

- [Why two layers](#why-two-layers)
- [Full dual-layer example](#full-dual-layer-example)
- [What each layer buys you](#what-each-layer-buys-you)

## Why two layers

Hydra configs use OmegaConf under the hood, which provides basic type checking but lacks the rich validation that Pydantic offers. The recommended pattern is to define Hydra-compatible dataclasses for the config store, then convert to Pydantic models at runtime for full validation.

## Full dual-layer example

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig
from pydantic import BaseModel, Field
import hydra


# Pydantic model for runtime validation
class TrainingConfig(BaseModel):
    """Training configuration with validation."""
    learning_rate: float = Field(gt=0, default=1e-3)
    batch_size: int = Field(ge=1, default=32)
    epochs: int = Field(ge=1, default=100)
    optimizer: str = Field(default="adamw")
    weight_decay: float = Field(ge=0, default=0.01)


class DataConfig(BaseModel):
    """Data configuration."""
    data_dir: str
    train_split: float = Field(gt=0, lt=1, default=0.8)
    val_split: float = Field(gt=0, lt=1, default=0.1)
    test_split: float = Field(gt=0, lt=1, default=0.1)
    num_workers: int = Field(ge=0, default=4)
    pin_memory: bool = True


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    training: TrainingConfig
    data: DataConfig
    seed: int = Field(ge=0, default=42)
    experiment_name: str = Field(min_length=1)


# Hydra dataclass for config store (structured configs)
@dataclass
class TrainingHydraConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adamw"
    weight_decay: float = 0.01


@dataclass
class DataHydraConfig:
    data_dir: str = MISSING
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ExperimentHydraConfig:
    training: TrainingHydraConfig = TrainingHydraConfig()
    data: DataHydraConfig = DataHydraConfig()
    seed: int = 42
    experiment_name: str = MISSING


# Register with ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=ExperimentHydraConfig)


def hydra_to_pydantic(cfg: DictConfig) -> ExperimentConfig:
    """Convert Hydra config to Pydantic for validation."""
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(cfg, resolve=True)
    return ExperimentConfig(**raw)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra."""
    # Validate with Pydantic
    config = hydra_to_pydantic(cfg)

    # Use validated config
    train(config)
```

## What each layer buys you

The dual-layer approach gives you the best of both worlds: Hydra handles composition and CLI overrides, while Pydantic enforces constraints like `learning_rate > 0` or `batch_size >= 1` at runtime.

Use `MISSING` in the Hydra dataclasses for values that must be supplied (by a config group, an experiment file, or the CLI) rather than inventing a dummy default that could silently produce a wrong run.

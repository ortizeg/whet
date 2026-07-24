# Hydra Config

The Hydra Config skill covers hierarchical configuration management for ML experiments using Hydra and OmegaConf, with structured configs and experiment overrides.

**Skill directory:** `skills/hydra-config/`

## Purpose

ML experiments involve dozens of hyperparameters, data paths, and infrastructure settings. Hardcoding them leads to untrackable experiments. This skill teaches Claude Code to use Hydra for composable YAML configurations with proper schema validation, command-line overrides, and multi-run sweeps. Every experiment becomes reproducible through its configuration alone.

## When to Use

- Any project with configurable hyperparameters
- Experiment management with parameter sweeps
- Projects that need different configs for dev, staging, and production
- Multi-model or multi-dataset projects with shared infrastructure config

## Key Patterns

### Directory Structure

```
configs/
    config.yaml           # Root config, composes groups
    model/
        resnet.yaml       # Model-specific params
        efficientnet.yaml
    data/
        imagenet.yaml     # Dataset-specific params
        coco.yaml
    trainer/
        default.yaml      # Trainer settings
        fast_dev.yaml     # Quick validation run
    experiment/
        baseline.yaml     # Experiment overrides
        ablation_lr.yaml
```

### Root Configuration

```yaml
# configs/config.yaml
defaults:
  - model: resnet
  - data: imagenet
  - trainer: default
  - _self_

seed: 42
output_dir: ${hydra:runtime.cwd}/outputs
```

### Structured Config with Pydantic

```python
from __future__ import annotations

from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 1000
    dropout: float = 0.1

@dataclass
class TrainerConfig:
    max_epochs: int = 100
    precision: str = "16-mixed"
    accelerator: str = "auto"
    devices: int = 1

cs = ConfigStore.instance()
cs.store(name="model_schema", node=ModelConfig, group="model")
cs.store(name="trainer_schema", node=TrainerConfig, group="trainer")
```

### Hydra Entry Point

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model = build_model(cfg.model)
    data = build_datamodule(cfg.data)
    trainer = build_trainer(cfg.trainer)
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
```

## Anti-Patterns to Avoid

- Do not use `OmegaConf.to_container(cfg, resolve=True)` everywhere -- keep configs as DictConfig for interpolation support
- Do not hardcode paths in configs -- use Hydra interpolations like `${hydra:runtime.cwd}`
- Do not put logic in config files -- configs are data, not code
- Avoid deeply nested configs with more than 3 levels

## Combines Well With

- **PyTorch Lightning** -- Configure Trainer, model, and data through Hydra
- **W&B / MLflow** -- Log resolved configs as experiment parameters
- **Pydantic** -- Structured configs with validation
- **Testing** -- Test config composition and resolution

## Full Reference

See [`skills/hydra-config/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/hydra-config/SKILL.md) for patterns including multi-run sweeps, Hydra callbacks, and integration with Optuna for hyperparameter optimization.

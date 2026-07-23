---
name: hydra-config
description: >
  Use this skill when managing complex, hierarchical experiment configuration with Hydra
  — structured/dataclass configs, config groups and composition, command-line overrides,
  multi-run sweeps, and validating the composed config with Pydantic. Reach for it any
  time an experiment has many swappable config pieces (model/data/augmentation) and you'd
  otherwise juggle argparse flags, even if the user doesn't say "Hydra". For plain data
  or settings validation without config composition, see pydantic.
---

# Hydra Configuration Skill

Use Hydra for managing complex, hierarchical configurations in machine learning and computer vision projects. This skill covers structured configs, config composition, command-line overrides, multi-run sweeps, and integration with Pydantic validation.

## Why Hydra

Managing configuration in ML projects is notoriously difficult. Experiments require dozens of parameters across data loading, model architecture, training schedules, augmentation pipelines, and evaluation settings. Hardcoded values lead to unmaintainable code, and ad-hoc config dictionaries lack type safety and validation.

Hydra solves these problems by providing:

- **Hierarchical configuration composition** -- build complex configs from smaller, reusable pieces
- **Command-line overrides** -- change any parameter without editing files
- **Multi-run (sweep) support** -- run hyperparameter searches with a single command
- **Plugin ecosystem** -- integrate with launchers, sweepers, and logging frameworks
- **Automatic working directory management** -- each run gets its own output directory
- **Config interpolation** -- reference other config values with `${}` syntax

## Integration with Pydantic

Hydra configs use OmegaConf under the hood, which provides basic type checking but lacks the rich validation that Pydantic offers. The recommended pattern is to define Hydra-compatible dataclasses for the config store, then convert to Pydantic models at runtime for full validation.

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

The dual-layer approach gives you the best of both worlds: Hydra handles composition and CLI overrides, while Pydantic enforces constraints like `learning_rate > 0` or `batch_size >= 1` at runtime.

## Directory Structure

Organize configuration files by concern. Each subdirectory represents a config group that can be swapped via the defaults list or CLI overrides.

```
conf/
├── config.yaml          # Default config (top-level)
├── training/
│   ├── default.yaml
│   ├── fast.yaml        # Quick training for debugging
│   └── full.yaml        # Full training run
├── data/
│   ├── coco.yaml
│   ├── imagenet.yaml
│   └── custom.yaml
├── model/
│   ├── resnet50.yaml
│   ├── efficientnet.yaml
│   └── yolov8.yaml
├── augmentation/
│   ├── basic.yaml
│   ├── heavy.yaml
│   └── none.yaml
└── experiment/
    ├── debug.yaml
    └── production.yaml
```

## Config Files

### conf/config.yaml

The top-level config uses the `defaults` list to compose from config groups. The `_self_` entry controls where this file's values are placed relative to the defaults.

```yaml
defaults:
  - training: default
  - data: coco
  - model: resnet50
  - augmentation: basic
  - _self_

seed: 42
experiment_name: "default_experiment"

# Interpolation example
output_dir: "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
```

### conf/training/default.yaml

```yaml
learning_rate: 1e-3
batch_size: 32
epochs: 100
optimizer: adamw
weight_decay: 0.01
scheduler: cosine
warmup_epochs: 5
gradient_clip_val: 1.0
```

### conf/training/fast.yaml

```yaml
learning_rate: 1e-3
batch_size: 64
epochs: 10
optimizer: adam
weight_decay: 0.0
scheduler: none
warmup_epochs: 0
gradient_clip_val: null
```

### conf/training/full.yaml

```yaml
learning_rate: 3e-4
batch_size: 16
epochs: 300
optimizer: adamw
weight_decay: 0.05
scheduler: cosine
warmup_epochs: 10
gradient_clip_val: 1.0
```

### conf/data/coco.yaml

```yaml
data_dir: "/data/coco"
train_split: 0.8
val_split: 0.1
test_split: 0.1
num_workers: 8
pin_memory: true
image_size: 640
format: "coco"
```

### conf/data/imagenet.yaml

```yaml
data_dir: "/data/imagenet"
train_split: 0.9
val_split: 0.05
test_split: 0.05
num_workers: 16
pin_memory: true
image_size: 224
format: "imagefolder"
```

### conf/model/resnet50.yaml

```yaml
name: resnet50
pretrained: true
num_classes: 80
dropout: 0.1
freeze_backbone: false
backbone_lr_factor: 0.1
```

### conf/augmentation/heavy.yaml

```yaml
horizontal_flip: 0.5
vertical_flip: 0.0
rotation_limit: 30
brightness_limit: 0.3
contrast_limit: 0.3
hue_shift_limit: 20
mosaic_prob: 0.5
mixup_prob: 0.3
cutout_prob: 0.2
```

## Command-Line Overrides

Hydra allows overriding any config value from the command line without editing files.

```bash
# Override single value
python train.py training.learning_rate=1e-4

# Override multiple values
python train.py training.learning_rate=1e-4 training.batch_size=64 seed=123

# Use different config group
python train.py training=fast data=imagenet

# Multi-run sweep over learning rates
python train.py --multirun training.learning_rate=1e-3,1e-4,1e-5

# Multi-run sweep over multiple parameters (grid)
python train.py --multirun \
    training.learning_rate=1e-3,1e-4 \
    training.batch_size=16,32,64

# Override nested values
python train.py model.num_classes=10 data.image_size=320

# Set experiment name
python train.py experiment_name="lr_sweep_v2"
```

## Advanced Patterns

### Config Groups with Package Directive

When you need to mount a config group at a specific path in the config tree, use the `@package` directive.

```yaml
# conf/server/apache.yaml
# @package _group_
host: localhost
port: 8080
```

### Recursive Defaults

Compose configs that themselves reference other defaults.

```yaml
# conf/experiment/production.yaml
defaults:
  - /training: full
  - /data: imagenet
  - /model: efficientnet
  - /augmentation: heavy

seed: 0
experiment_name: "production_run"
```

Then run with:

```bash
python train.py +experiment=production
```

### Instantiate Pattern

Use `hydra.utils.instantiate` to create objects directly from config. This is powerful for swapping model architectures, optimizers, or schedulers without changing code.

```python
from hydra.utils import instantiate
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Config specifies _target_ for instantiation
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
```

With corresponding config:

```yaml
# conf/optimizer/adamw.yaml
_target_: torch.optim.AdamW
lr: 1e-3
weight_decay: 0.01
betas: [0.9, 0.999]

# conf/scheduler/cosine.yaml
_target_: torch.optim.lr_scheduler.CosineAnnealingLR
T_max: 100
eta_min: 1e-6
```

### Variable Interpolation

Reference other config values using `${}` syntax. This eliminates duplication and keeps values consistent.

```yaml
training:
  epochs: 100
  warmup_epochs: 5

scheduler:
  _target_: torch.optim.lr_scheduler.CosineAnnealingLR
  T_max: ${training.epochs}

output_dir: "outputs/${experiment_name}/${now:%Y-%m-%d}"
checkpoint_dir: "${output_dir}/checkpoints"
log_dir: "${output_dir}/logs"
```

### Environment Variable Resolution

Read values from environment variables with fallbacks.

```yaml
data:
  data_dir: ${oc.env:DATA_DIR,/default/data/path}
  cache_dir: ${oc.env:CACHE_DIR,/tmp/cache}

wandb:
  api_key: ${oc.env:WANDB_API_KEY}
  project: ${oc.env:WANDB_PROJECT,my-project}
```

## Integration with Lightning

Hydra integrates cleanly with PyTorch Lightning by passing validated config objects into Lightning modules and data modules.

```python
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = hydra_to_pydantic(cfg)

    model = MyModel(config.model)
    datamodule = MyDataModule(config.data)

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=config.training.gradient_clip_val,
        default_root_dir=cfg.output_dir,
    )
    trainer.fit(model, datamodule)
```

## Integration with Experiment Tracking

Log the resolved Hydra config to your experiment tracker so every run is fully reproducible.

```python
from omegaconf import OmegaConf
import wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert to dict for logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Log to W&B
    wandb.init(
        project=cfg.experiment_name,
        config=config_dict,
    )

    # Also save as YAML artifact
    with open("config_resolved.yaml", "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    wandb.save("config_resolved.yaml")
```

## Best Practices

1. **Always validate with Pydantic** -- Hydra configs lack runtime validation constraints; use Pydantic to enforce value ranges and types at startup before any training begins.

2. **Use structured configs** -- Define dataclasses for type-safe config definitions and register them with the ConfigStore. This catches typos and type errors early.

3. **Compose configs by concern** -- Split configs into groups (training, data, model, augmentation) so they can be mixed and matched independently.

4. **Use the defaults list** -- Compose from multiple config files rather than writing monolithic configs. This makes experiments reproducible by specifying which config variant was used.

5. **Override from CLI, never hardcode** -- Every parameter should be overridable from the command line. Never write `if debug: lr = 0.01` in your code.

6. **Log the resolved config** -- Save the fully resolved configuration with every experiment run so you can reproduce it exactly.

7. **Use MISSING for required values** -- Mark values that must be provided with `MISSING` instead of using dummy defaults that might silently produce wrong results.

8. **Pin Hydra version** -- Hydra's behavior can change between versions. Pin the version in your dependencies to avoid surprises.

9. **Keep experiment configs** -- Store named experiment configs (e.g., `experiment/ablation_v3.yaml`) that combine specific config groups for reproducibility.

10. **Use interpolation for derived values** -- If `checkpoint_dir` depends on `output_dir`, use `${output_dir}/checkpoints` rather than duplicating the path.

## Common Pitfalls

- **Forgetting `_self_`** -- Without `_self_` in the defaults list, the order of config merging may surprise you. Always include it explicitly.
- **Mutating OmegaConf objects** -- OmegaConf DictConfigs are not regular dicts. Convert to a container or Pydantic model before mutating.
- **Relative paths** -- Hydra changes the working directory by default. Use absolute paths or `hydra.runtime.cwd` to resolve relative paths correctly.
- **Missing `version_base`** -- Always set `version_base=None` (or a specific version) in `@hydra.main` to avoid deprecation warnings and ensure consistent behavior.

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

Use Hydra to compose ML/CV experiment configuration from small, swappable config
groups, override any value from the command line, and sweep with `--multirun`.
This page holds the everyday core — a `conf/` tree, a `defaults` list, and a
`@hydra.main` entry point. The deep dives below carry structured configs,
composition rules, sweeps, instantiation, and framework integrations.

## Why Hydra

Managing configuration in ML projects is notoriously difficult. Experiments require dozens of parameters across data loading, model architecture, training schedules, augmentation pipelines, and evaluation settings. Hardcoded values lead to unmaintainable code, and ad-hoc config dictionaries lack type safety and validation.

Hydra solves these problems by providing:

- **Hierarchical configuration composition** -- build complex configs from smaller, reusable pieces
- **Command-line overrides** -- change any parameter without editing files
- **Multi-run (sweep) support** -- run hyperparameter searches with a single command
- **Plugin ecosystem** -- integrate with launchers, sweepers, and logging frameworks
- **Automatic working directory management** -- each run gets its own output directory
- **Config interpolation** -- reference other config values with `${}` syntax

## Core pattern: config groups + `@hydra.main`

Organize configuration by concern. Each subdirectory of `conf/` is a config group
whose variants can be swapped from the defaults list or the CLI.

```
conf/
├── config.yaml          # Top-level config
├── training/            # default.yaml, fast.yaml, full.yaml
├── data/                # coco.yaml, imagenet.yaml, custom.yaml
├── model/               # resnet50.yaml, efficientnet.yaml, yolov8.yaml
├── augmentation/        # basic.yaml, heavy.yaml, none.yaml
└── experiment/          # debug.yaml, production.yaml
```

The top-level `conf/config.yaml` picks one variant per group. `_self_` controls
where this file's own values land relative to the composed defaults.

```yaml
defaults:
  - training: default
  - data: coco
  - model: resnet50
  - augmentation: basic
  - _self_

seed: 42
experiment_name: "default_experiment"
output_dir: "outputs/${experiment_name}/${now:%Y-%m-%d_%H-%M-%S}"
```

A group file is plain YAML — `conf/training/default.yaml`:

```yaml
learning_rate: 1e-3
batch_size: 32
epochs: 100
optimizer: adamw
weight_decay: 0.01
scheduler: cosine
```

Entry point: decorate `main` with `@hydra.main`, then validate the composed
config with Pydantic before anything expensive runs.

```python
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    seed: int = Field(ge=0, default=42)
    experiment_name: str = Field(min_length=1)
    # ... nested training/data/model models


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = ExperimentConfig(**OmegaConf.to_container(cfg, resolve=True))
    train(config)


if __name__ == "__main__":
    main()
```

Everyday CLI usage:

```bash
python train.py training.learning_rate=1e-4        # override a value
python train.py training=fast data=imagenet        # swap config groups
python train.py --multirun training.learning_rate=1e-3,1e-4,1e-5  # sweep
```

## Conventions

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

## Anti-patterns

- **Forgetting `_self_`** -- Without `_self_` in the defaults list, the order of config merging may surprise you. Always include it explicitly.
- **Mutating OmegaConf objects** -- OmegaConf DictConfigs are not regular dicts. Convert to a container or Pydantic model before mutating.
- **Relative paths** -- Hydra changes the working directory by default. Use absolute paths or `hydra.runtime.cwd` to resolve relative paths correctly.
- **Missing `version_base`** -- Always set `version_base=None` (or a specific version) in `@hydra.main` to avoid deprecation warnings and ensure consistent behavior.

## Deep dives

- `references/structured-configs-and-pydantic.md` — read when defining dataclass configs, registering with the ConfigStore, or wiring the Hydra→Pydantic validation layer.
- `references/config-groups-and-composition.md` — read when laying out `conf/`, writing group YAML files, using `@package`, or building recursive experiment configs.
- `references/overrides-and-multirun.md` — read when running parameter sweeps, multirun jobs, or needing exact CLI override syntax (`+`, `++`, group swaps).
- `references/instantiate-and-interpolation.md` — read when building objects from `_target_` config, using `${}` interpolation, or reading env vars into config.
- `references/lightning-and-tracking-integration.md` — read when passing a Hydra config into PyTorch Lightning or logging the resolved config to W&B/MLflow.

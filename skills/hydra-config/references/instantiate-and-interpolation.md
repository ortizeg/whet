# Instantiate and Interpolation

Scope: building objects directly from config with `hydra.utils.instantiate`, referencing other config values with `${}`, and reading environment variables.

## Contents

- [Instantiate pattern](#instantiate-pattern)
- [Variable interpolation](#variable-interpolation)
- [Environment variable resolution](#environment-variable-resolution)

## Instantiate pattern

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

## Variable interpolation

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

## Environment variable resolution

Read values from environment variables with fallbacks.

```yaml
data:
  data_dir: ${oc.env:DATA_DIR,/default/data/path}
  cache_dir: ${oc.env:CACHE_DIR,/tmp/cache}

wandb:
  api_key: ${oc.env:WANDB_API_KEY}
  project: ${oc.env:WANDB_PROJECT,my-project}
```

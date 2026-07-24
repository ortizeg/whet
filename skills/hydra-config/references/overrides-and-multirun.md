# Command-Line Overrides and Multirun Sweeps

Scope: changing any config value from the command line, swapping config groups, and running grid sweeps with `--multirun`.

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

Notes:

- `key=value` overrides an existing key; `+key=value` adds a new one; `++key=value` adds or overrides.
- `--multirun` (`-m`) takes the cartesian product of every comma-separated list, so two lists of 2 and 3 values launch 6 runs.
- Each multirun job gets its own output directory under `multirun/<date>/<time>/<job-num>/`.
- Named experiment configs are added with `+experiment=production` because `experiment` is not in the base defaults list.

# Hydra Configuration Skill

## Purpose

This skill provides patterns for managing complex, hierarchical configurations in ML and CV projects using Hydra. It covers structured configs, config composition, CLI overrides, multi-run sweeps, and Pydantic integration for runtime validation.

## When to Use

- You are building an ML/CV project with many configurable parameters (learning rate, batch size, model architecture, augmentation pipeline, data paths, etc.)
- You need to run experiments with different configurations without editing code
- You want to sweep over hyperparameters from the command line
- You need reproducible experiment configs that can be logged and versioned

## Key Patterns

- **Dual-layer config**: Define Hydra dataclasses for the config store, then convert to Pydantic models at runtime for validation
- **Config composition**: Split configs by concern (training, data, model, augmentation) and compose them via the defaults list
- **Instantiate pattern**: Use `hydra.utils.instantiate` to create objects (optimizers, schedulers, models) directly from config
- **Variable interpolation**: Reference other config values with `${}` syntax to avoid duplication

## Usage

```bash
# Basic run
python train.py

# Override parameters
python train.py training.learning_rate=1e-4 data=imagenet

# Hyperparameter sweep
python train.py --multirun training.learning_rate=1e-3,1e-4,1e-5
```

## Benefits

- Eliminates hardcoded parameters scattered throughout code
- Makes every experiment reproducible via saved config files
- Enables rapid experimentation through CLI overrides
- Composes complex configs from simple, reusable pieces
- Validates all parameters at startup before expensive training begins

## See Also

- `SKILL.md` in this directory for full documentation and code examples
- `pydantic` skill for validation patterns
- `pytorch-lightning` skill for training loop integration

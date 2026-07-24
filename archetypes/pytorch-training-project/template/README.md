# ${project_name}

${description}

## Setup

Environments are managed with [pixi](https://pixi.sh):

```bash
pixi install            # create the environment from pixi.toml
pixi shell              # activate it (or prefix commands with `pixi run`)
```

Add dependencies with pixi — never `pip install`:

```bash
pixi add --pypi timm            # runtime dependency
pixi add --feature dev pytest   # dev-only dependency
```

## Training

Configuration is composed by [Hydra](https://hydra.cc) from `configs/` and validated
with Pydantic before training starts. Any value is overridable from the CLI.

```bash
# Default training
python -m ${package_name}.train

# Swap a config group and override a value
python -m ${package_name}.train model=resnet50 trainer.max_epochs=50

# Debug run (single batch, CPU)
python -m ${package_name}.train trainer=debug

# Hyperparameter sweep
python -m ${package_name}.train --multirun model.learning_rate=1e-3,1e-4
```

Hydra writes each run's outputs (resolved config, logs, checkpoints) to `outputs/`.

## Development

```bash
pytest
ruff check .
ruff format .
mypy src/ --strict
```

The same commands are available as pixi tasks: `pixi run test`, `pixi run lint`,
`pixi run format`, `pixi run typecheck`, `pixi run train`, `pixi run debug`.

## Project Structure

```
${project_slug}/
├── src/${package_name}/
│   ├── __init__.py
│   ├── model.py          # LightningModule (Classifier)
│   ├── data.py           # LightningDataModule (ImageDataModule)
│   ├── train.py          # Hydra entry point + validated configs
│   └── transforms.py     # Data augmentations
├── configs/
│   ├── config.yaml       # Main Hydra config (defaults list)
│   ├── model/            # default.yaml, resnet50.yaml
│   ├── data/             # default.yaml
│   └── trainer/          # default.yaml, debug.yaml
├── tests/
│   ├── conftest.py       # Shared fixtures
│   ├── test_model.py
│   ├── test_data.py
│   └── test_train.py     # Config composition + fast_dev_run smoke test
├── pixi.toml             # Environment and tasks
├── pyproject.toml        # Package metadata and tool config
└── README.md
```

## Conventions

- **Batches are `(images, labels)` tuples.** `ImageDataModule` yields them and
  `Classifier.training_step` / `validation_step` unpack them. Keep that contract
  when swapping in a real dataset.
- **Logging uses Loguru** (`from loguru import logger`) — never `print` or
  `logging.getLogger`.
- **Configs are Pydantic V2 models** (`ModelConfig`, `DataConfig`, `TrainerConfig`,
  `ExperimentConfig`); Hydra composes the YAML, Pydantic validates it.
- **src-layout**: importable code lives under `src/${package_name}/`.

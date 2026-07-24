# PyTorch Training Project Archetype

A complete project template for training deep learning models with PyTorch Lightning. This archetype provides a production-ready scaffold that enforces best practices in experiment management, configuration, type safety, and reproducibility. It is designed for computer vision practitioners who need a structured, extensible foundation for training workflows ranging from simple classifiers to complex multi-task models.

## Purpose

The PyTorch Training Project archetype solves the recurring problem of bootstrapping ML training codebases from scratch. Every new training project faces the same set of infrastructure decisions: how to manage configurations, structure data loading, organize model definitions, handle logging, set up testing, and integrate CI/CD. This archetype encodes opinionated answers to all of those decisions so that you can focus on the modeling work itself.

The archetype is built around three pillars: PyTorch Lightning for structured training loops, Hydra for hierarchical configuration management, and Pydantic for runtime config validation. Together these ensure that experiments are reproducible, configurations are validated before training begins, and the training loop is cleanly separated from research logic.

## Use Cases

- **Image classification training** -- Train ResNet, EfficientNet, or custom backbones on labeled image datasets with built-in augmentation pipelines.
- **Object detection model development** -- Structure YOLO, Faster R-CNN, or SSD training with proper anchor generation and loss computation.
- **Semantic segmentation training** -- Build U-Net, DeepLab, or Mask2Former training pipelines with per-pixel loss functions and IoU metrics.
- **Transfer learning experiments** -- Fine-tune pretrained models on domain-specific data with frozen backbone scheduling and discriminative learning rates.
- **Hyperparameter optimization** -- Integrate with Optuna or Ray Tune through Hydra's sweeper plugins for systematic hyperparameter search.
- **Multi-GPU and distributed training** -- Leverage Lightning's built-in DDP, FSDP, and DeepSpeed strategies without modifying training code.

## Directory Structure

The rendered project (`${package_name}` is derived from the project name):

```
${project_slug}/
├── .gitignore
├── pixi.toml                        # Environment, dependencies, and tasks
├── pyproject.toml                   # Package metadata + ruff/mypy/pytest config
├── README.md                        # Generated project README
├── configs/                         # Hydra configuration tree
│   ├── config.yaml                  # Root config (defaults list + seed)
│   ├── model/
│   │   ├── default.yaml             # resnet18 backbone
│   │   └── resnet50.yaml            # Alternate backbone preset
│   ├── data/
│   │   └── default.yaml             # Data loading config
│   └── trainer/
│       ├── default.yaml             # Standard training settings
│       └── debug.yaml               # Single-batch CPU smoke run
├── src/${package_name}/
│   ├── __init__.py
│   ├── model.py                     # Classifier LightningModule + ModelConfig
│   ├── data.py                      # ImageDataModule + DataConfig
│   ├── transforms.py                # torchvision transform pipelines
│   └── train.py                     # Hydra entry point, TrainerConfig, ExperimentConfig
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Shared fixtures
    ├── test_model.py                # Forward pass + train/val step contracts
    ├── test_data.py                 # Dataloader batch contract
    └── test_train.py                # Config composition + fast_dev_run smoke test
```

Everything above is a starting point: grow `model.py`/`data.py` into packages
(`models/`, `data/`) once the project outgrows single modules.

## Key Features

- **PyTorch Lightning** for structured, boilerplate-free training loops with automatic mixed precision, gradient accumulation, and distributed training.
- **Hydra** for hierarchical configuration management with command-line overrides, config composition, and multirun sweeps.
- **Pydantic V2** for strict runtime validation of every configuration value before training begins, catching typos and out-of-range values early.
- **Loguru** as the single logging convention — no `print`, no `logging.getLogger`.
- **Pixi** as the environment manager, with tasks for train, test, lint, format, and typecheck.
- **Type safety** with mypy strict mode enabled by default.
- **Tested by default** — the generated project ships tests that cover the batch contract, the config tree, and an end-to-end `fast_dev_run`.
- **Experiment tracking** integration points for Weights and Biases, MLflow, or TensorBoard (opt-in via Lightning loggers).

## Configuration Variables

These are the variables the scaffold engine substitutes into template files:

| Variable | Description | Default |
|---|---|---|
| `project_name` | Human-readable project name displayed in docs and logs | Required |
| `project_slug` | URL-safe directory and distribution name | Derived from `project_name` |
| `package_name` | Python import name (underscored, PEP 8 compliant) | Derived from `project_slug` |
| `description` | One-line project description | Empty |
| `author` | Author name for `pyproject.toml` / `pixi.toml` | Empty |
| `python_version` | Minimum Python version constraint | 3.11 |

## Dependencies

Declared in the generated `pixi.toml` (and mirrored in `[project].dependencies`):

```toml
[pypi-dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
lightning = ">=2.2"
torchmetrics = ">=1.3"
hydra-core = ">=1.3"
pydantic = ">=2.6"
loguru = ">=0.7"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
```

## Usage

### Project Initialization

```bash
whet init pytorch-training-project --name yolo-detector

cd yolo-detector
pixi install
```

### Training

```bash
# Run training with the default configuration
python -m ${package_name}.train

# Override specific config values from the command line
python -m ${package_name}.train model.learning_rate=1e-4 trainer.max_epochs=50

# Swap a config group preset
python -m ${package_name}.train model=resnet50
python -m ${package_name}.train trainer=debug

# Run a Hydra multirun sweep
python -m ${package_name}.train --multirun model.learning_rate=1e-3,1e-4,1e-5
```

Equivalent pixi tasks: `pixi run train`, `pixi run debug`.

### Testing and Quality

```bash
pytest                    # or: pixi run test
pytest --cov=src/
ruff check .              # or: pixi run lint
mypy src/ --strict        # or: pixi run typecheck
```

## Customization Guide

### Adding a New Model Architecture

1. Extend `ModelConfig` in `src/${package_name}/model.py` with the new architecture hyperparameters (use `pydantic.Field` constraints).
2. Teach `Classifier._build_backbone` how to construct it, or split backbone construction into its own module once there is more than one.
3. Add a Hydra YAML file in `configs/model/` (e.g. `efficientnet.yaml`) with the full set of `ModelConfig` keys.
4. Add a test in `tests/test_model.py` verifying forward-pass shapes, and one in `tests/test_train.py` verifying the new config group composes.
5. Select it from the CLI with `model=efficientnet` — no code changes to the entry point.

### Adding New Data Sources

1. Implement a `torch.utils.data.Dataset` returning `(image, label)` tuples — the contract `training_step` and `validation_step` unpack.
2. Instantiate it in `ImageDataModule.setup` in place of `torchvision.datasets.FakeData`.
3. Extend `DataConfig` with the paths, split ratios, and preprocessing options it needs, and mirror them in `configs/data/default.yaml`.
4. Extend `src/${package_name}/transforms.py` with the augmentation pipeline for the new data.

### Adding Custom Callbacks

1. Create a callback inheriting from `lightning.pytorch.Callback`.
2. Instantiate it in `build_trainer` in `src/${package_name}/train.py`, driven by a new config group (e.g. `configs/callbacks/`).
3. Common additions include early stopping, model checkpointing, learning rate monitors, and prediction visualizers.

### Enabling Experiment Tracking

Lightning loggers are wired in `build_trainer` in `src/${package_name}/train.py`. Supported integrations include Weights and Biases (`WandbLogger`), MLflow (`MLFlowLogger`), and TensorBoard (`TensorBoardLogger`). Give each logger its own Hydra config group so the choice is a command-line override rather than a code edit.

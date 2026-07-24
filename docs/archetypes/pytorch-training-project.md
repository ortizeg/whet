# PyTorch Training Project

End-to-end model training with PyTorch Lightning, Hydra configuration management, and experiment tracking.

## Purpose

This archetype provides a complete training pipeline structure for computer vision models. It includes LightningModule and LightningDataModule patterns, Hydra-based configuration, experiment logging, checkpointing, and CI/CD workflows for automated training validation.

## Directory Structure

```
{{project_slug}}/
├── configs/
│   ├── config.yaml            # Main Hydra config
│   ├── model/
│   │   ├── resnet.yaml
│   │   └── efficientnet.yaml
│   ├── data/
│   │   └── coco.yaml
│   └── trainer/
│       ├── default.yaml
│       └── debug.yaml
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── train.py               # Hydra entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract model interface
│   │   └── detector.py        # Concrete LightningModule
│   ├── data/
│   │   ├── __init__.py
│   │   └── datamodule.py      # LightningDataModule
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   └── test_data.py
└── ...                        # Common base files
```

## Key Components

- **LightningModule** with Pydantic config for all hyperparameters
- **LightningDataModule** with configurable augmentations
- **Hydra configs** for experiment composition
- **Experiment tracking** via W&B, MLflow, or TensorBoard (opt-in)
- **Model checkpointing** with configurable strategies

## Usage

```bash
# Train with default config
python -m my_project.train

# Override parameters
python -m my_project.train model=efficientnet trainer.max_epochs=50

# Debug mode (1 batch, no logging)
python -m my_project.train trainer=debug
```

## Customization

- Add new model architectures in `src/{{package_name}}/models/`
- Add dataset configs in `configs/data/`
- Extend augmentation pipelines in the DataModule
- Configure experiment tracking by selecting optional skills (wandb, mlflow, tensorboard)

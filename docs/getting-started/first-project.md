# First Project: Object Detection Training Pipeline

This walkthrough builds a complete object detection training pipeline from scratch, demonstrating how multiple skills compose together to produce a production-grade project.

## Project Goal

We will build a YOLO-style object detection trainer with:

- PyTorch Lightning training loop with mixed-precision support
- Hydra-based configuration for experiments
- W&B experiment tracking
- Docker container for reproducible training
- Pre-commit hooks and CI/CD

## Step 1: Initialize the Project

Create a new project directory and initialize it:

```bash
mkdir object-detector && cd object-detector
pixi init --name object-detector
```

## Step 2: Define the Project with an Archetype

Use the PyTorch Training archetype as the foundation:

```bash
claude "Create an object detection training project using the
       pytorch-training-project archetype. The detector should support
       COCO-format datasets and output bounding boxes with class labels."
```

This generates the base project structure following the archetype's conventions.

## Step 3: Build the Model with PyTorch Lightning Skill

```bash
claude "Using the pytorch-lightning skill, create a LightningModule for
       object detection. Include:
       - A backbone (ResNet-50 FPN) with pretrained weights
       - A detection head with classification and regression branches
       - Training step with focal loss + smooth L1 loss
       - Validation step with mAP calculation
       - Learning rate scheduling with cosine annealing
       - Proper logging of loss components and metrics"
```

The skill ensures Claude generates code with:

- Proper `__init__`, `forward`, `training_step`, `validation_step`, and `configure_optimizers` methods
- Type hints on all parameters and return values
- Metric computation separated from the training loop
- Checkpoint-friendly state management

## Step 4: Create the Data Pipeline

```bash
claude "Using the pytorch-lightning skill, create a LightningDataModule for
       COCO-format object detection datasets. Include:
       - Configurable train/val/test splits
       - Albumentations-based augmentations (horizontal flip, color jitter,
         random crop with bbox-safe transforms)
       - Proper collate function for variable-size bounding boxes
       - DataLoader with configurable num_workers and pin_memory"
```

## Step 5: Add Configuration Management

```bash
claude "Using the hydra-config skill, create structured configuration for this
       object detection project. Include:
       - Main config composing model, data, trainer, and logger configs
       - Model config with backbone choice, num_classes, and head parameters
       - Data config with dataset paths, augmentation parameters, batch size
       - Trainer config with max_epochs, precision, accelerator, devices
       - Experiment override configs for small/medium/large model variants"
```

Expected output structure:

```
configs/
    config.yaml
    model/
        detector.yaml
    data/
        coco.yaml
    trainer/
        default.yaml
        fast_dev.yaml
    experiment/
        small.yaml
        medium.yaml
        large.yaml
```

## Step 6: Integrate Experiment Tracking

```bash
claude "Using the wandb skill, add Weights & Biases experiment tracking:
       - Log training/validation metrics per step and epoch
       - Log sample predictions as W&B image tables every N steps
       - Save model checkpoints as W&B artifacts
       - Define a hyperparameter sweep config for learning rate and batch size
       - Add a W&B alert for when validation mAP plateaus"
```

## Step 7: Containerize for Reproducibility

```bash
claude "Using the docker-cv skill, create a Dockerfile for this project:
       - Multi-stage build (builder + runtime)
       - CUDA 12.1 + cuDNN 8 base image
       - Pixi-based dependency installation
       - Non-root user for security
       - Health check endpoint
       - Volume mounts for data and checkpoints
       Also create a docker-compose.yml for local multi-GPU training."
```

## Step 8: Add Code Quality and Testing

```bash
claude "Using the code-quality, testing, and pre-commit skills:
       - Configure ruff for linting and formatting
       - Set up mypy with strict mode
       - Create pre-commit hooks for ruff, mypy, and pytest
       - Write unit tests for the model (forward pass shape checks)
       - Write unit tests for the data module (batch shape, augmentation)
       - Write an integration test for a single training step"
```

## Step 9: Set Up CI/CD

```bash
claude "Using the github-actions skill, create CI workflows:
       - Lint and type-check on every PR
       - Run tests with a small synthetic dataset
       - Build and push Docker image on merge to main
       - Optional: trigger a training run on a self-hosted GPU runner"
```

## Final Project Structure

```
object-detector/
    .github/
        workflows/
            ci.yml
            docker.yml
    configs/
        config.yaml
        model/
            detector.yaml
        data/
            coco.yaml
        trainer/
            default.yaml
            fast_dev.yaml
        experiment/
            small.yaml
            medium.yaml
            large.yaml
    src/
        object_detector/
            __init__.py
            model.py
            data.py
            transforms.py
            losses.py
            metrics.py
            train.py
            predict.py
    tests/
        conftest.py
        test_model.py
        test_data.py
        test_transforms.py
        test_integration.py
    Dockerfile
    docker-compose.yml
    pixi.toml
    pyproject.toml
    .pre-commit-config.yaml
    sweep.yaml
```

## Step 10: Train

```bash
# Local training
uv run python src/object_detector/train.py experiment=small

# Docker training
docker compose up train

# Multi-GPU training
uv run python src/object_detector/train.py trainer.devices=4 trainer.strategy=ddp
```

## Key Takeaways

1. **Start with an archetype** to get the right project structure from the beginning.
2. **Layer skills incrementally** -- add configuration before experiment tracking, add testing before CI/CD.
3. **Each skill produces consistent output** because it follows the same conventions every time Claude generates code.
4. **Skills compose naturally** -- the Hydra config skill knows how to configure a Lightning trainer, the W&B skill knows how to integrate with Lightning loggers, and the Docker skill knows how to package a Pixi-managed project.

## Next Steps

- Explore individual [skill documentation](../skills/index.md) for deeper dives
- Browse the [skills overview](../skills/index.md) to add more capabilities
- Check out [common patterns](../examples/common-patterns.md) for more examples

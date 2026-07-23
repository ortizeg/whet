---
name: mlflow
description: >
  Use this skill whenever the project needs experiment tracking, a model registry, or
  model serving with MLflow — logging params, metrics, and artifacts, comparing runs,
  versioning and promoting models through stages, and running a self-hosted tracking
  server. Reach for it any time you'd otherwise hand-roll run tracking or ask "which
  config produced this result?" and the stack is MLflow, even if the user doesn't say it.
  For the W&B or TensorBoard alternatives see wandb and tensorboard.
---

# MLflow Tracking Integration for ML Projects

## Overview

MLflow is an open-source platform for managing the full machine learning lifecycle. It provides experiment tracking, model packaging, a model registry, and model serving capabilities. MLflow is opt-in: it should only be used when the developer explicitly enables it, and all code should function without it installed. Unlike W&B, MLflow can run entirely self-hosted, making it suitable for projects with strict data privacy requirements.

## Why Use MLflow

MLflow addresses several key challenges in ML development:

- **Experiment tracking** with parameters, metrics, and artifacts logged to a central store.
- **Model packaging** in a standard format that can be deployed anywhere.
- **Model registry** for managing model versions and deployment stages.
- **Self-hosted option** for organizations that cannot use cloud services.
- **Language-agnostic design** supporting Python, R, Java, and REST APIs.
- **Open source** with no vendor lock-in.

## Setup and Configuration

### Installation

```bash
# Using pip
pip install mlflow

# Using pixi
pixi add mlflow --feature experiment-tracking

# With extras for specific backends
pip install mlflow[extras]
```

### Local Tracking Server

For local development, MLflow stores data in the `mlruns` directory by default:

```bash
# Start the MLflow UI (local tracking)
mlflow ui --port 5000

# Access at http://localhost:5000
```

### Remote Tracking Server

For team collaboration, set up a remote tracking server:

```bash
# Start server with PostgreSQL backend and S3 artifact store
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost:5432/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Configure the client to point to the remote server:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
```

Or use an environment variable:

```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## Experiment Tracking

### Basic Experiment Setup

```python
import mlflow

# Set or create an experiment
mlflow.set_experiment("object-detection")

# Start a run
with mlflow.start_run(run_name="yolov8-baseline"):
    # Log parameters
    mlflow.log_param("model", "yolov8")
    mlflow.log_param("learning_rate", 1e-3)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("image_size", 640)

    # Train model
    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_map = evaluate(model, val_loader)

        # Log metrics with step
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_mAP", val_map, step=epoch)

    # Log final metrics
    mlflow.log_metric("best_mAP", best_map)
```

### Using Pydantic Config with MLflow

```python
from pydantic import BaseModel, Field
import mlflow

class TrainingConfig(BaseModel):
    """Training configuration."""
    model_name: str = "yolov8"
    learning_rate: float = Field(gt=0, default=1e-3)
    batch_size: int = Field(ge=1, default=32)
    epochs: int = Field(ge=1, default=100)
    image_size: int = Field(ge=32, default=640)
    optimizer: str = "AdamW"

def log_pydantic_config(config: TrainingConfig) -> None:
    """Log all fields of a Pydantic model as MLflow parameters."""
    for key, value in config.model_dump().items():
        mlflow.log_param(key, value)

# Usage
config = TrainingConfig(learning_rate=5e-4)
with mlflow.start_run():
    log_pydantic_config(config)
```

### Opt-in Pattern

```python
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

@contextmanager
def optional_mlflow_run(
    use_mlflow: bool,
    experiment_name: str,
    run_name: str,
) -> Generator[Any, None, None]:
    """Context manager that optionally creates an MLflow run."""
    if not use_mlflow:
        yield None
        return

    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except ImportError:
        print("mlflow not installed. Skipping experiment tracking.")
        yield None

# Usage
with optional_mlflow_run(use_mlflow=True, experiment_name="detection", run_name="exp-001") as run:
    if run is not None:
        import mlflow
        mlflow.log_param("model", "yolov8")
```

## Parameter and Metric Logging

### Batch Parameter Logging

```python
import mlflow

# Log multiple parameters at once
params = {
    "model": "yolov8",
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealing",
    "warmup_epochs": 5,
}
mlflow.log_params(params)

# Log multiple metrics at once
metrics = {
    "train_loss": 0.25,
    "val_loss": 0.30,
    "val_mAP": 0.45,
    "val_mAP50": 0.62,
}
mlflow.log_metrics(metrics, step=epoch)
```

### Logging Tags

```python
import mlflow

# Set tags for organization and filtering
mlflow.set_tag("task", "object_detection")
mlflow.set_tag("dataset", "coco_2017")
mlflow.set_tag("gpu", "A100")
mlflow.set_tag("framework", "pytorch")

# Set multiple tags at once
mlflow.set_tags({
    "developer": "team-cv",
    "priority": "high",
    "stage": "experiment",
})
```

## Artifact Storage

### Logging Files

```python
import mlflow

# Log a single file
mlflow.log_artifact("checkpoints/best_model.pt", artifact_path="models")

# Log an entire directory
mlflow.log_artifacts("outputs/predictions/", artifact_path="predictions")

# Log a text file
with open("training_summary.txt", "w") as f:
    f.write(f"Best mAP: {best_map:.4f}\nBest epoch: {best_epoch}")
mlflow.log_artifact("training_summary.txt")
```

### Logging Figures

```python
import mlflow
import matplotlib.pyplot as plt

# Create a training curve plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Training Curves")

# Log the figure
mlflow.log_figure(fig, "plots/training_curves.png")
plt.close(fig)
```

### Logging Dictionaries and Tables

```python
import mlflow

# Log a dictionary as JSON
results = {
    "mAP": 0.45,
    "mAP50": 0.62,
    "mAP75": 0.38,
    "per_class": {"car": 0.52, "person": 0.48, "bike": 0.35},
}
mlflow.log_dict(results, "results/eval_metrics.json")

# Log a table
table = {
    "columns": ["class", "AP", "AP50", "AP75"],
    "data": [
        ["car", 0.52, 0.71, 0.45],
        ["person", 0.48, 0.65, 0.40],
        ["bike", 0.35, 0.50, 0.28],
    ],
}
mlflow.log_table(data=table, artifact_file="results/per_class_ap.json")
```

## Model Registry

The model registry provides a centralized model store with versioning and stage transitions.

### Registering a Model

```python
import mlflow
import mlflow.pytorch

# Log and register model in one step
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log PyTorch model and register it
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="yolov8-coco",
    )
```

### Managing Model Versions

```python
from mlflow import MlflowClient

client = MlflowClient()

# Transition model to staging
client.transition_model_version_stage(
    name="yolov8-coco",
    version=3,
    stage="Staging",
)

# Promote to production
client.transition_model_version_stage(
    name="yolov8-coco",
    version=3,
    stage="Production",
)

# Add description
client.update_model_version(
    name="yolov8-coco",
    version=3,
    description="Best model with mAP=0.52 on COCO val2017",
)
```

### Loading a Registered Model

```python
import mlflow.pytorch

# Load by stage
model = mlflow.pytorch.load_model("models:/yolov8-coco/Production")

# Load by version
model = mlflow.pytorch.load_model("models:/yolov8-coco/3")

# Load by run ID
model = mlflow.pytorch.load_model("runs:/abc123def456/model")
```

## Serving Models

MLflow can serve models as REST APIs:

```bash
# Serve a registered model
mlflow models serve \
    --model-uri "models:/yolov8-coco/Production" \
    --port 8000 \
    --no-conda

# Make predictions
curl -X POST http://localhost:8000/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [...]}'
```

## Integration with PyTorch Lightning

MLflow integrates with PyTorch Lightning through `MLFlowLogger`:

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

# Create MLflow logger
logger = MLFlowLogger(
    experiment_name="object-detection",
    run_name="lightning-experiment-001",
    tracking_uri="http://mlflow-server:5000",
    log_model=True,
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="auto",
)

# Train
trainer.fit(model, datamodule=dm)

# Access run info
print(f"Run ID: {logger.run_id}")
print(f"Experiment ID: {logger.experiment_id}")
```

## Comparison with W&B

| Feature | MLflow | W&B |
|---------|--------|-----|
| Self-hosted | Yes (primary mode) | Yes (enterprise) |
| Cloud-hosted | Community Edition | Yes (free tier) |
| Experiment tracking | Yes | Yes |
| Model registry | Yes | Yes |
| Hyperparameter sweeps | No (use Optuna) | Yes (built-in) |
| Interactive dashboards | Basic | Advanced |
| Image/video logging | Limited | Excellent |
| Artifact management | Yes | Yes |
| Model serving | Yes (built-in) | No |
| Open source | Yes (Apache 2.0) | Partially |
| Offline mode | Default | Yes |

**Choose MLflow when**: Self-hosting is required, model serving is needed, or vendor independence is important.

**Choose W&B when**: Interactive visualization, image/video logging, or hyperparameter sweeps are priorities.

## Local vs Remote Tracking Server

### Local (Default)

```python
# Data stored in ./mlruns directory
import mlflow
mlflow.set_tracking_uri("file:///path/to/mlruns")
```

Best for: Individual development, quick experiments, offline work.

### Remote

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

Best for: Team collaboration, centralized experiment management, production workflows.

## Best Practices

1. **Use experiments** to group related runs (one experiment per project or task).
2. **Name runs descriptively** to make them easy to find later.
3. **Log all hyperparameters** at the start of every run.
4. **Use tags** for filtering: task type, dataset, GPU, developer.
5. **Log artifacts** for model checkpoints, predictions, and evaluation results.
6. **Use the model registry** to manage model lifecycle (staging, production).
7. **Set `MLFLOW_TRACKING_URI`** as an environment variable rather than hardcoding.
8. **Add `.mlruns/` to `.gitignore`** to keep local tracking data out of version control.
9. **Use `mlflow.autolog()`** for quick setup with supported frameworks.
10. **Wrap MLflow calls** behind feature flags for opt-in behavior.

```python
# Autolog for PyTorch
import mlflow
mlflow.pytorch.autolog()

with mlflow.start_run():
    trainer.fit(model, datamodule=dm)
```

## Summary

MLflow is a comprehensive, open-source ML lifecycle management platform. Its self-hosted nature, model serving capabilities, and vendor independence make it an excellent choice for teams that need full control over their experiment tracking infrastructure. When used alongside or as an alternative to W&B, it provides robust experiment tracking, artifact management, and model registry functionality with no cloud dependency required.

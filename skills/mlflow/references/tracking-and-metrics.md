# Tracking Runs, Parameters, Metrics, and Tags

Scope: structuring experiments and runs, logging params/metrics/tags, driving logging from a Pydantic config, and keeping MLflow strictly opt-in.

## Contents

- [Basic experiment setup](#basic-experiment-setup)
- [Using a Pydantic config with MLflow](#using-a-pydantic-config-with-mlflow)
- [Opt-in pattern](#opt-in-pattern)
- [Batch parameter and metric logging](#batch-parameter-and-metric-logging)
- [Logging tags](#logging-tags)

## Basic experiment setup

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

## Using a Pydantic config with MLflow

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

## Opt-in pattern

MLflow is opt-in: it should only be used when the developer explicitly enables it,
and all code must function without it installed.

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

## Batch parameter and metric logging

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

## Logging tags

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

# MLflow

The MLflow skill covers ML lifecycle management, experiment tracking, model registry, and model serving using MLflow in CV/ML projects.

**Skill directory:** `skills/mlflow/`

## Purpose

MLflow provides a self-hosted alternative to cloud-based experiment trackers, with a strong model registry and serving component. This skill teaches Claude Code to integrate MLflow with PyTorch Lightning for experiment tracking, register and version models, create reproducible ML pipelines, and serve models via REST API. It is particularly suited for organizations that need on-premise experiment tracking.

## When to Use

- Organizations that require self-hosted experiment tracking (compliance, data privacy)
- Projects that need a model registry with staging/production transitions
- Pipelines that benefit from MLflow's Projects and Recipes abstractions
- Teams migrating from ad-hoc model management to structured MLOps

## Key Patterns

### Lightning Integration

```python
from __future__ import annotations

import lightning as L
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

logger = MLFlowLogger(
    experiment_name="object-detection",
    tracking_uri="http://localhost:5000",
    log_model=True,
)

trainer = L.Trainer(
    max_epochs=100,
    logger=logger,
)
```

### Model Registration

```python
import mlflow.pytorch

def register_model(
    model: L.LightningModule,
    model_name: str,
    run_id: str,
) -> None:
    """Register a trained model in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name=model_name)
```

### Experiment Logging

```python
with mlflow.start_run(run_name="resnet50-baseline") as run:
    mlflow.log_params({
        "backbone": "resnet50",
        "learning_rate": 1e-3,
        "batch_size": 32,
    })

    # Training loop
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader)
        val_metrics = validate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_mAP": val_metrics["mAP"],
        }, step=epoch)

    # Log the model
    mlflow.pytorch.log_model(model, "model")
```

## Anti-Patterns to Avoid

- Do not use the default `mlruns/` local directory in production -- configure a proper tracking URI
- Do not register models without validation metrics -- always log metrics alongside the model
- Avoid logging large artifacts (datasets, raw images) as run artifacts -- keep them in cloud object storage and log a reference
- Do not mix MLflow and W&B in the same project unless migrating

## Combines Well With

- **PyTorch Lightning** -- MLFlowLogger integrates with Lightning Trainer
- **Hydra Config** -- Log resolved configs as run parameters
- **Docker CV** -- MLflow tracking server in a container

## Full Reference

See [`skills/mlflow/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/mlflow/SKILL.md) for patterns including model serving, A/B testing with model stages, and custom MLflow plugins for CV-specific artifact types.

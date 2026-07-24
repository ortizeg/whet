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

MLflow is an open-source platform for managing the full machine learning lifecycle:
experiment tracking, model packaging, a model registry, and model serving. It is
opt-in — it should only be used when the developer explicitly enables it, and all
code should function without it installed. Unlike W&B, MLflow can run entirely
self-hosted, making it suitable for projects with strict data privacy requirements.

## Why Use MLflow

MLflow addresses several key challenges in ML development:

- **Experiment tracking** with parameters, metrics, and artifacts logged to a central store.
- **Model packaging** in a standard format that can be deployed anywhere.
- **Model registry** for managing model versions and deployment stages.
- **Self-hosted option** for organizations that cannot use cloud services.
- **Language-agnostic design** supporting Python, R, Java, and REST APIs.
- **Open source** with no vendor lock-in.

## Core pattern: an experiment, a run, and logged values

Point the client at a tracking server (env var preferred), name an experiment, and
wrap training in a run. Params are logged once; metrics are logged per step.

```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000   # omit for local ./mlruns
mlflow ui --port 5000                                   # local UI
```

```python
import mlflow

mlflow.set_experiment("object-detection")

with mlflow.start_run(run_name="yolov8-baseline"):
    mlflow.log_params({
        "model": "yolov8",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
    })
    mlflow.set_tags({"task": "object_detection", "dataset": "coco_2017"})

    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_map = evaluate(model, val_loader)
        mlflow.log_metrics({"train_loss": train_loss, "val_mAP": val_map}, step=epoch)

    mlflow.log_artifact("checkpoints/best_model.pt", artifact_path="models")
    mlflow.log_metric("best_mAP", best_map)
```

Keep MLflow optional by guarding the import and the run behind a flag, so the
training script still runs on a machine where MLflow is not installed:

```python
from contextlib import contextmanager


@contextmanager
def optional_mlflow_run(use_mlflow: bool, experiment_name: str, run_name: str):
    if not use_mlflow:
        yield None
        return
    try:
        import mlflow
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except ImportError:
        yield None
```

## Conventions

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

## Anti-patterns

- **Making MLflow a hard dependency** — importing `mlflow` at module top level breaks every script for anyone who has not installed it. Import inside the guarded helper.
- **Hardcoding the tracking URI** in application code — the same script must run locally and against the team server without edits.
- **Logging metrics without `step`** — a metric history with no step axis cannot be plotted against epochs or compared across runs.
- **Logging hyperparameters after training** — log params first so a crashed run still records what it was trying.
- **Deploying from a run URI** — pin deployments to `models:/name/Production`, not `runs:/<id>/model`, so promotion does not require a code change.
- **Committing `mlruns/`** — local tracking data in git bloats the repo and conflicts constantly.

## Deep dives

- `references/self-hosted-setup.md` — read when installing MLflow, standing up a tracking server, or choosing between local `mlruns` and a remote backend.
- `references/tracking-and-metrics.md` — read when logging params/metrics/tags in detail, driving logging from a Pydantic config, or implementing the full opt-in wrapper.
- `references/artifacts.md` — read when logging files, directories, matplotlib figures, JSON dicts, or tables to a run.
- `references/model-registry.md` — read when registering models, promoting versions through Staging/Production, or loading a registered model.
- `references/serving.md` — read when exposing a model as a REST endpoint with `mlflow models serve`.
- `references/lightning-and-autolog.md` — read when wiring `MLFlowLogger` into a Lightning `Trainer` or enabling framework autologging.
- `references/mlflow-vs-wandb.md` — read when deciding which experiment-tracking stack the project should adopt.

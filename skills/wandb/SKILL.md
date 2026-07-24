---
name: wandb
description: >
  Use this skill when experiment tracking and collaboration run on Weights & Biases —
  logging metrics, managing artifacts, running hyperparameter sweeps, the model registry,
  dataset versioning, and opt-in graceful degradation. Reach for it any time training runs
  need hosted tracking, sweeps, or team sharing and the stack is W&B, even if the user
  just says "track this experiment" or "log the run". For a zero-setup local dashboard see
  tensorboard; for the MLflow alternative see mlflow.
---

# Weights & Biases (W&B) Integration for ML Projects

W&B is a hosted experiment tracking platform for ML: metrics and dashboards, versioned
dataset/model artifacts, hyperparameter sweeps, a model registry, and team reports.
It is **opt-in** — enable it explicitly, and keep code working when it is absent.
Setup is `pip install wandb` (or `pixi add wandb --feature experiment-tracking`) plus
`wandb login` / `WANDB_API_KEY`.

## Essential Core

Almost every use is `wandb.init(...)` once with a config, then `wandb.log({...})` in the loop.

```python
import wandb

run = wandb.init(
    project="my-cv-project",
    name="yolov8-experiment-001",
    config={"model": "yolov8", "learning_rate": 1e-3, "batch_size": 32, "epochs": 100},
    tags=["baseline", "yolov8", "coco"],
    notes="Baseline YOLOv8 training on COCO subset",
)

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_map = evaluate(model, val_loader)

    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mAP": val_map,
        "epoch": epoch,
    })

run.finish()
```

Keep the dependency soft — import inside a guard so the project runs without W&B:

```python
def create_logger(use_wandb: bool, project: str, config: dict):
    if not use_wandb:
        return None
    try:
        import wandb
        return wandb.init(project=project, config=config)
    except ImportError:
        print("wandb not installed. Skipping experiment tracking.")
        return None
```

Debug without uploading anything, then backfill later:

```bash
export WANDB_MODE=offline
wandb sync ./wandb/offline-run-*
```

## What to Reach For

| Need | Call | Detail in |
|------|------|-----------|
| Scalar metrics | `wandb.log({"train/loss": ...})` | `references/logging-metrics-and-media.md` |
| Images, boxes, masks, video | `wandb.Image` / `wandb.Video` | `references/logging-metrics-and-media.md` |
| Gradients and parameters | `wandb.watch(model, log="all")` | `references/logging-metrics-and-media.md` |
| Versioned checkpoints/datasets | `wandb.Artifact` + `log_artifact` | `references/artifacts.md` |
| Hyperparameter search | `wandb.sweep` + `wandb.agent` | `references/sweeps.md` |
| Promote a model | `run.link_artifact(...)` | `references/model-registry.md` |
| Lightning training | `WandbLogger` | `references/opt-in-integration.md` |
| Notification on finish | `wandb.alert(...)` | `references/logging-metrics-and-media.md` |

## Conventions

1. **Always use the `config` parameter** in `wandb.init()` to log all hyperparameters.
2. **Use structured metric names** with prefixes: `train/loss`, `val/mAP`, `test/precision`.
3. **Log images sparingly** (every N epochs) to avoid slowing down training.
4. **Use tags** to categorize runs: `["baseline", "augmented", "ablation"]`.
5. **Set `WANDB_MODE=offline`** for debugging without uploading data.
6. **Wrap W&B calls** in try/except or behind feature flags for robustness.
7. **Use artifacts** for all datasets and model checkpoints to ensure reproducibility.
8. **Write notes** when starting a run to document the hypothesis being tested.
9. **Create W&B Reports** for sharing results with the team or in papers.
10. **Use `wandb.alert()`** to get notified when training finishes or metrics degrade.
11. **Pass Pydantic configs as `config.model_dump()`** rather than hand-built dicts.
12. **Mirror the repo structure** in W&B: `team / project / run`.

## Anti-Patterns

- ❌ Making `wandb` a hard import at module top level — breaks the project for anyone without it.
- ❌ Flat metric names (`loss`, `map`) — no train/val separation in the dashboard.
- ❌ Logging images or video every step — uploads dominate training time.
- ❌ Re-running `wandb.init()` inside a training loop instead of once per run.
- ❌ Storing checkpoints only on local disk — use artifacts so runs stay reproducible.
- ❌ Untagged, unnamed runs — impossible to find later.
- ❌ Leaving hyperparameters out of `config` — the run cannot be reproduced or swept over.
- ❌ Committing the API key — use `wandb login` locally and CI secrets in pipelines.

## Deep dives

- `references/logging-metrics-and-media.md` — read when logging scalars beyond the basics, learning rate/gradients, custom step axes, bounding boxes, segmentation masks, video, or alerts.
- `references/artifacts.md` — read when versioning model checkpoints or datasets, or downloading a previously logged artifact.
- `references/sweeps.md` — read when running hyperparameter sweeps or Bayesian search.
- `references/model-registry.md` — read when promoting a model artifact to the registry or aliasing production versions.
- `references/opt-in-integration.md` — read when setting up auth/CI, wiring `WandbLogger` into PyTorch Lightning, using Pydantic configs, or running offline.

## Summary

W&B brings order to ML experimentation: log metrics, visualize results, manage artifacts, and
optimize hyperparameters so teams iterate faster and make data-driven decisions. Its opt-in
nature ensures it integrates without imposing a hard dependency on the project.

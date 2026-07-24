---
name: tensorboard
description: >
  Use this skill when logging and visualizing PyTorch training locally with TensorBoard —
  scalar metrics, image logging, model-graph visualization, histogram tracking,
  profiling, and zero-setup local experiment monitoring. Reach for it any time you want a
  quick local training dashboard without a hosted service, even if the user just says
  "let me watch the loss" or "log the metrics somewhere". For hosted, collaborative
  tracking with sweeps and a model registry, see wandb or mlflow.
---

# TensorBoard Logging for ML Projects

TensorBoard is the lightweight, local visualization toolkit for PyTorch training: no account,
no server, no internet — it reads event files straight off disk. It covers scalars, images,
histograms, computation graphs, embeddings, and profiling, and is PyTorch Lightning's default
logger. Treat it as **opt-in**: code must still run when it is absent.

## Essential Core

Create a `SummaryWriter`, call `add_scalar` in the loop, and close it.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_map = evaluate(model, val_loader)

    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/mAP", val_map, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

writer.close()  # or writer.flush() to keep logging
```

Then watch it live:

```bash
tensorboard --logdir=logs/            # http://localhost:6006
tensorboard --logdir=logs/exp1:baseline,logs/exp2:augmented   # compare runs
```

For computer vision, the other call you reach for constantly is an image grid:

```python
from torchvision.utils import make_grid

grid = make_grid(images[:16], nrow=4, normalize=True, padding=2)
writer.add_image("batch/inputs", grid, epoch)
```

Keep the dependency soft so the project runs without TensorBoard installed:

```python
class TensorBoardLogger:
    """Optional TensorBoard logger."""

    def __init__(self, log_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("tensorboard not installed. Skipping logging.")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
```

## What to Reach For

| Need | Call | Detail in |
|------|------|-----------|
| Metrics over time | `add_scalar` / `add_scalars` | `references/scalars-and-metrics.md` |
| Compare hyperparameters | `add_hparams` | `references/scalars-and-metrics.md` |
| Images, grids, figures | `add_image` / `add_figure` | `references/image-logging.md` |
| Weight/gradient distributions | `add_histogram` | `references/model-graph-and-histograms.md` |
| Model architecture | `add_graph` | `references/model-graph-and-histograms.md` |
| Text, embeddings, PR curves | `add_text` / `add_embedding` / `add_pr_curve` | `references/model-graph-and-histograms.md` |
| Find bottlenecks | `torch.profiler` + `tensorboard_trace_handler` | `references/profiling.md` |
| Lightning training | `TensorBoardLogger` | `references/setup-and-integration.md` |

## Conventions

1. **Use consistent tag naming**: `{split}/{metric}` (e.g., `train/loss`, `val/mAP`).
2. **Log images periodically** (every 5-10 epochs) to avoid large log files.
3. **Use `flush()` or `close()`** to ensure data is written to disk.
4. **Organize log directories** by experiment: `logs/{experiment_name}/{run_name}`.
5. **Log hyperparameters** with `add_hparams` for comparison across runs.
6. **Profile early** to catch data loading bottlenecks before long training runs.
7. **Use `add_scalars`** (plural) to overlay related metrics on the same chart.
8. **Add `.gitignore` entries** for log directories: `logs/`, `runs/`, `lightning_logs/`.
9. **Wrap TensorBoard** behind an opt-in flag for flexibility.
10. **Combine with Lightning** for automatic metric logging with minimal code.

## Anti-Patterns

- ❌ Leaving the writer unclosed — buffered events never reach disk and the run looks empty.
- ❌ Logging images or histograms every step — event files balloon and training slows.
- ❌ Flat tag names (`loss`, `map`) — TensorBoard cannot group them into charts.
- ❌ Writing every run into one `log_dir` — runs overwrite and become uncomparable.
- ❌ Committing `logs/` or `runs/` to git.
- ❌ Making `torch.utils.tensorboard` a hard import — keep it behind the opt-in wrapper.
- ❌ Reaching for TensorBoard when you need artifact versioning, sweeps, or a model registry — use wandb or mlflow.

## Deep dives

- `references/scalars-and-metrics.md` — read when logging metrics beyond the basics, overlaying series with `add_scalars`, or comparing runs with the HParams plugin.
- `references/image-logging.md` — read when logging single images, grids, or matplotlib figures such as detection overlays.
- `references/model-graph-and-histograms.md` — read when visualizing the model graph, tracking weight/gradient/activation distributions, or writing text, embedding, or PR-curve summaries.
- `references/profiling.md` — read when hunting training bottlenecks with the PyTorch Profiler trace handler.
- `references/setup-and-integration.md` — read when installing/launching TensorBoard, wiring the opt-in logger, using `TensorBoardLogger` with PyTorch Lightning, or viewing logs from a remote machine.

## Summary

TensorBoard is the simplest, most lightweight experiment visualization tool available for
PyTorch projects. Its deep integration with PyTorch (through `SummaryWriter`) and PyTorch
Lightning (as the default logger) makes it the go-to choice for local development and quick
experiments. For team collaboration or advanced features like artifact management and sweeps,
supplement it with W&B or MLflow.

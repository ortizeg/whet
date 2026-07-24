---
name: matplotlib
description: >
  Use this skill when creating plots and figures for ML/CV work with Matplotlib —
  training curves, confusion matrices, image grids, bounding-box overlays, feature-map
  and t-SNE/UMAP embedding visualizations, and publication-quality figures. Reach for it
  any time you'd otherwise hand-tweak matplotlib to visualize model results, metrics, or
  images, even if the user just says "plot the loss" or "show these predictions". For a
  live training dashboard inside a tracker, see tensorboard or wandb.
---

# Matplotlib Skill

Visualization patterns for ML/CV: training curves, confusion matrices, image
grids, detection overlays, feature maps, embedding plots, publication figures,
and experiment-tracker logging. Use matplotlib when you need precise control over
image annotations, multi-panel figures, and non-interactive server rendering.

## Essential Core

Set the backend and a project-wide style once at import time, then build plotting functions
that take arrays and **return** a `plt.Figure` so callers decide whether to save, log, or
further customize it.

```python
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive; set before pyplot use in scripts

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox_inches": "tight",
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "legend.fontsize": 10,
    "legend.framealpha": 0.8,
})
```

Training curves are the plot you will write first on every project — loss on a log scale, one
extra panel per additional metric:

```python
from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_metrics: dict[str, list[float]] | None = None,
    val_metrics: dict[str, list[float]] | None = None,
    save_path: str | Path | None = None,
    title: str = "Training Progress",
) -> plt.Figure:
    """Plot loss (log scale) plus one panel per additional metric. Returns the figure."""
    num_metrics = 1 + (len(train_metrics) if train_metrics else 0)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss", color="#2196F3")
    axes[0].plot(epochs, val_losses, label="Val Loss", color="#FF5722")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_yscale("log")  # loss curves read best on log scale

    if train_metrics and val_metrics:
        for i, (name, train_vals) in enumerate(train_metrics.items(), 1):
            val_vals = val_metrics.get(name, [])
            axes[i].plot(epochs, train_vals, label=f"Train {name}", color="#2196F3")
            if val_vals:
                axes[i].plot(epochs, val_vals, label=f"Val {name}", color="#FF5722")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(name)
            axes[i].set_title(name)
            axes[i].legend()

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Conventions

1. **Always close figures** -- `plt.close(fig)` after saving in training loops to free memory.
2. **Use the Agg backend** on servers to avoid display-related errors.
3. **Save PDF/SVG for papers**, PNG for dashboards -- vector scales without pixelation.
4. **Use `tight_layout()`** to prevent labels being cut off.
5. **Return figures** (`plt.Figure`) so callers can further customize or save.
6. **Separate data from presentation** -- compute metrics first, pass arrays to plotting functions.
7. **Use consistent colors** -- define a project palette and reuse it across plots.
8. **Label everything** -- axes, legends, titles, units. Use log scale for loss curves.
9. **Normalize the axes array** with `np.atleast_2d(axes)` in grid plots so single-row and
   single-cell layouts index the same way.
10. **Detach and move tensors to CPU** (`.detach().cpu().numpy()`) before plotting them.

## Anti-Patterns

- **Never call `plt.show()` inside library code** — return the figure and let the caller decide.
- **Never leave figures open in a loop** — the global registry keeps them alive and leaks memory.
- **Never rely on the default interactive backend on a server** — set `Agg` before importing pyplot.
- **Never truncate a bar-chart y-axis** — anchor at zero; truncated axes exaggerate differences.
- **Never plot GPU tensors directly** — detach and convert to numpy first.
- **Never use `jet`** — prefer perceptually uniform colormaps (`viridis`) for magnitude data.
- **Never scale a figure in LaTeX** — size it correctly at creation so fonts stay at the intended size.

## Deep dives

- `references/confusion-matrices.md` — read when plotting classification confusion matrices, normalized or raw.
- `references/image-grids-and-overlays.md` — read when displaying batches of images, augmentation previews, or drawing detection boxes over an image.
- `references/feature-maps-and-embeddings.md` — read when visualizing CNN activations via forward hooks, or t-SNE/UMAP projections of embeddings.
- `references/publication-figures.md` — read when producing paper figures: single-column sizing, serif fonts, vector PDF, grouped comparison bar charts.
- `references/saving-and-tracker-integration.md` — read when saving to multiple formats, using a `.mplstyle` file, or logging figures to W&B/MLflow.

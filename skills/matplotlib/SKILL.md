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

## Style Configuration

Set a consistent project-wide style in code, or load a `.mplstyle` file with
`plt.style.use("path/to/custom.mplstyle")`. Use the `Agg` backend on servers.

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

```ini
# custom.mplstyle
figure.figsize: 10, 6
figure.dpi: 150
savefig.dpi: 300
savefig.bbox_inches: tight
font.size: 12
axes.grid: True
grid.alpha: 0.3
lines.linewidth: 2
```

## Training Curve Plotting

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

## Confusion Matrix Visualization

```python
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | Path | None = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot an annotated confusion matrix, optionally row-normalized. Returns the figure."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # handle classes with zero samples

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Image Grid Display

Works for augmentation previews, sample batches, etc. Grayscale images (2D) are
detected automatically.

```python
def plot_image_grid(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    ncols: int = 4,
    figsize_per_image: tuple[float, float] = (3, 3),
    save_path: str | Path | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """Display images (H,W,3 or H,W) in a grid. Returns the figure."""
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    figsize = (figsize_per_image[0] * ncols, figsize_per_image[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < n:
            cmap = "gray" if images[idx].ndim == 2 else None
            ax.imshow(images[idx], cmap=cmap)
            if titles:
                ax.set_title(titles[idx], fontsize=10)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Bounding Box Overlay

Matplotlib overlays are better than OpenCV for notebooks and papers (crisp text,
vector output).

```python
import matplotlib.patches as patches


def plot_detections(
    image: np.ndarray,
    boxes: np.ndarray,             # (N, 4) xyxy
    labels: list[str],
    scores: np.ndarray | None = None,
    class_colors: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Draw detection boxes with labels/scores over an RGB image. Returns the figure."""
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    default_colors = plt.cm.tab10.colors

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        if class_colors and label in class_colors:
            color = class_colors[label]
        else:
            color = default_colors[hash(label) % len(default_colors)]

        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        ))
        text = f"{label} {scores[i]:.2f}" if scores is not None else label
        ax.text(
            x1, y1 - 5, text,
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
        )

    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Feature Map Visualization

```python
import torch


def plot_feature_maps(
    feature_map: torch.Tensor,     # (C, H, W) or (1, C, H, W)
    num_channels: int = 16,
    ncols: int = 8,
    save_path: str | Path | None = None,
    title: str = "Feature Maps",
) -> plt.Figure:
    """Visualize the first N channels of a CNN feature map. Returns the figure."""
    if feature_map.dim() == 4:
        feature_map = feature_map[0]

    fm = feature_map.detach().cpu().numpy()
    num_channels = min(num_channels, fm.shape[0])
    nrows = (num_channels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        if idx < num_channels:
            ax.imshow(fm[idx], cmap="viridis")
            ax.set_title(f"Ch {idx}", fontsize=8)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# Capture activations via a forward hook, then plot:
activations: dict[str, torch.Tensor] = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

model.layer3.register_forward_hook(hook_fn("layer3"))
model(input_batch)
plot_feature_maps(activations["layer3"], save_path="feature_maps.png")
```

## t-SNE / UMAP Embedding Plots

```python
def plot_embeddings(
    embeddings: np.ndarray,        # (N, D)
    labels: np.ndarray,            # (N,)
    class_names: list[str] | None = None,
    method: str = "tsne",          # "tsne" or "umap"
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
    title: str | None = None,
    perplexity: int = 30,
) -> plt.Figure:
    """Reduce embeddings to 2D and scatter by class. Returns the figure."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names else str(label)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i]], label=name, s=10, alpha=0.7)

    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"{method.upper()} Embedding Visualization")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Publication-Quality Figures

Small serif fonts, thin lines, single-column width, PDF output.

```python
def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (3.5, 2.5),  # single-column width
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox_inches": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 8,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "legend.fontsize": 7,
        "legend.framealpha": 0.8,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.3,
        "text.usetex": False,
        "mathtext.fontset": "dejavuserif",
    })


def plot_comparison_bar_chart(
    methods: list[str],
    metrics: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing methods across metrics, with value labels."""
    setup_publication_style()
    n_metrics = len(metrics)
    x = np.arange(len(methods))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots()
    colors = plt.cm.Set2(np.linspace(0, 0.8, n_metrics))
    for i, (metric_name, values) in enumerate(metrics.items()):
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric_name, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig
```

## Saving and Experiment-Tracker Integration

Save vector (PDF/SVG) for papers and raster (PNG) for dashboards. Always
`plt.close(fig)` after logging inside a loop to free memory.

```python
def save_figure_multiformat(
    fig: plt.Figure,
    base_path: str | Path,
    formats: list[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in multiple formats. Returns saved paths."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(path)
    return saved


def log_figure_to_wandb(fig: plt.Figure, key: str, step: int | None = None) -> None:
    import wandb
    wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_figure_to_mlflow(fig: plt.Figure, artifact_path: str) -> None:
    import mlflow
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)
```

## Best Practices

1. **Always close figures** -- `plt.close(fig)` after saving in training loops to free memory.
2. **Use the Agg backend** on servers to avoid display-related errors.
3. **Save PDF/SVG for papers**, PNG for dashboards -- vector scales without pixelation.
4. **Use `tight_layout()`** to prevent labels being cut off.
5. **Return figures** (`plt.Figure`) so callers can further customize or save.
6. **Separate data from presentation** -- compute metrics first, pass arrays to plotting functions.
7. **Use consistent colors** -- define a project palette and reuse it across plots.
8. **Label everything** -- axes, legends, titles, units. Use log scale for loss curves.

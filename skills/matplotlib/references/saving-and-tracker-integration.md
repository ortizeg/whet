# Saving Figures and Experiment-Tracker Integration

Writing figures to disk in multiple formats, and pushing them into W&B or MLflow without leaking memory.

## Style Files

Instead of setting `plt.rcParams` in code, a project can ship a `.mplstyle` file and load it
with `plt.style.use("path/to/custom.mplstyle")`:

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

A style file is preferable once more than one entry point produces figures — it keeps the
project's look in one place and makes it overridable per-run.

## Multi-Format Saving

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
```

- Pass `base_path` **without** an extension; `with_suffix` supplies one per format.
- `bbox_inches="tight"` crops surrounding whitespace and, crucially, prevents axis labels and
  rotated tick labels from being cut off at the figure edge.
- `dpi` only affects raster formats; PDF and SVG ignore it.
- The `mkdir(parents=True, exist_ok=True)` means callers can point at a nested run directory
  that does not exist yet.

## Tracker Logging

```python
def log_figure_to_wandb(fig: plt.Figure, key: str, step: int | None = None) -> None:
    import wandb
    wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_figure_to_mlflow(fig: plt.Figure, artifact_path: str) -> None:
    import mlflow
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)
```

- **`plt.close(fig)` is the point of these wrappers.** Matplotlib keeps every unclosed figure
  alive in a global registry; a training loop that logs one figure per epoch without closing
  leaks memory until the process dies (and emits the "More than 20 figures have been opened"
  warning along the way).
- The tracker imports are function-local so the plotting module does not hard-depend on
  `wandb` or `mlflow` — matching the opt-in integration pattern used by the tracker skills.
- Pass `step` to W&B so figures land on the same x-axis as scalar metrics; MLflow keys the
  artifact by path instead, so include the epoch in `artifact_path` (e.g.
  `f"figures/confusion_matrix_epoch_{epoch}.png"`).
- `wandb.Image(fig)` rasterizes the figure; for a vector artifact, save a PDF first with
  `save_figure_multiformat` and log the file.

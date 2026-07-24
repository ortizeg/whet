"""Reusable plotting helpers for ${project_name}.

Every function takes an optional ``ax`` and returns the ``Axes`` it drew on, so
plots compose into subplot grids instead of each owning a figure. Nothing here
calls ``plt.show()`` — the notebook decides when to display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ${package_name}.data import class_counts

Features = NDArray[np.float64]
Labels = NDArray[np.int64]

FIGURE_STYLE: dict[str, Any] = {
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.25,
    "font.size": 10,
}


def use_project_style() -> None:
    """Apply the project-wide Matplotlib defaults.

    Call once, in the setup cell of a notebook, so every figure in the project
    looks the same without per-plot fiddling.
    """
    matplotlib.style.use(FIGURE_STYLE)
    logger.debug("Applied project Matplotlib style")


def _resolve_axes(ax: Axes | None, figsize: tuple[float, float]) -> Axes:
    """Return the caller's axes, or create a new figure and axes."""
    if ax is not None:
        return ax
    _, created = plt.subplots(figsize=figsize)
    return created


def plot_class_distribution(
    labels: Labels,
    ax: Axes | None = None,
    title: str = "Class distribution",
) -> Axes:
    """Draw a bar chart of samples per class."""
    counts = class_counts(labels)
    axes = _resolve_axes(ax, (5.0, 3.2))
    axes.bar([str(label) for label in counts], list(counts.values()), color="#4C72B0")
    axes.set_xlabel("Class")
    axes.set_ylabel("Samples")
    axes.set_title(title)
    return axes


def plot_scatter_2d(
    features: Features,
    labels: Labels,
    ax: Axes | None = None,
    title: str = "Feature space (first two dimensions)",
) -> Axes:
    """Scatter the first two feature dimensions, coloured by class."""
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        msg = f"features must be 2-D with at least 2 columns, got shape {array.shape}"
        raise ValueError(msg)
    target = np.asarray(labels).ravel()
    if target.size != array.shape[0]:
        msg = f"labels length {target.size} does not match {array.shape[0]} samples"
        raise ValueError(msg)

    axes = _resolve_axes(ax, (5.0, 4.4))
    for label in sorted({int(value) for value in target}):
        mask = target == label
        axes.scatter(array[mask, 0], array[mask, 1], s=12, alpha=0.75, label=f"class {label}")
    axes.set_xlabel("feature 0")
    axes.set_ylabel("feature 1")
    axes.set_title(title)
    axes.legend(frameon=False, fontsize=8)
    return axes


def save_figure(fig: Figure, path: Path, close: bool = True) -> Path:
    """Write ``fig`` to ``path`` and return it.

    Figures worth keeping go to ``outputs/figures/`` as real files. Committed
    notebooks keep their outputs stripped, so a saved PNG is the durable artifact.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    logger.info("Saved figure to {}", path)
    if close:
        plt.close(fig)
    return path

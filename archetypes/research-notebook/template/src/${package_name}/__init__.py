"""${project_name} — reusable research code.

Notebooks under ``notebooks/`` are for *exploration*. The moment a cell is worth
running twice, it moves here: typed, tested, and importable. Notebooks then read
like a narrative built out of these helpers instead of a pile of copy-paste.
"""

from __future__ import annotations

from ${package_name}.config import PATHS, ExperimentConfig, Paths
from ${package_name}.data import (
    class_counts,
    make_synthetic_dataset,
    split_indices,
)
from ${package_name}.viz import plot_class_distribution, plot_scatter_2d, save_figure

__version__ = "0.1.0"

__all__ = [
    "PATHS",
    "ExperimentConfig",
    "Paths",
    "class_counts",
    "make_synthetic_dataset",
    "plot_class_distribution",
    "plot_scatter_2d",
    "save_figure",
    "split_indices",
]

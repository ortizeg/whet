"""Tests for the reusable plotting helpers.

Plotting code is still code: it can raise, mislabel axes, or silently drop a
class. These tests assert on the returned ``Axes`` rather than on pixels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ${package_name}.data import Features, Labels, class_counts
from ${package_name}.viz import (
    FIGURE_STYLE,
    plot_class_distribution,
    plot_scatter_2d,
    save_figure,
    use_project_style,
)


def test_class_distribution_has_one_bar_per_class(dataset: tuple[Features, Labels]) -> None:
    _, labels = dataset
    axes = plot_class_distribution(labels)
    assert len(axes.patches) == len(class_counts(labels))
    assert axes.get_ylabel() == "Samples"


def test_class_distribution_uses_supplied_axes(dataset: tuple[Features, Labels]) -> None:
    _, labels = dataset
    _, provided = plt.subplots()
    returned = plot_class_distribution(labels, ax=provided)
    assert returned is provided


def test_scatter_draws_one_collection_per_class(dataset: tuple[Features, Labels]) -> None:
    features, labels = dataset
    axes = plot_scatter_2d(features, labels, title="custom")
    assert len(axes.collections) == len(class_counts(labels))
    assert axes.get_title() == "custom"


def test_scatter_rejects_one_dimensional_features() -> None:
    with pytest.raises(ValueError, match="at least 2 columns"):
        plot_scatter_2d(np.zeros((10, 1)), np.zeros(10, dtype=np.int64))


def test_scatter_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        plot_scatter_2d(np.zeros((10, 2)), np.zeros(5, dtype=np.int64))


def test_save_figure_writes_file(figures_dir: Path, dataset: tuple[Features, Labels]) -> None:
    features, labels = dataset
    axes = plot_scatter_2d(features, labels)
    figure = axes.get_figure()
    assert figure is not None
    target = save_figure(figure, figures_dir / "sub" / "scatter.png")
    assert target.is_file()
    assert target.stat().st_size > 0


def test_use_project_style_applies_rcparams() -> None:
    use_project_style()
    assert plt.rcParams["axes.grid"] == FIGURE_STYLE["axes.grid"]
    assert plt.rcParams["savefig.dpi"] == FIGURE_STYLE["savefig.dpi"]

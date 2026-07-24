"""Shared pytest fixtures for ${project_name}.

The code a notebook imports is ordinary Python, so it gets ordinary tests. If a
helper is too tangled to test here, it is too tangled to trust in a notebook.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import matplotlib
import pytest

# Must be selected before pyplot is imported anywhere: tests run headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from ${package_name}.config import ExperimentConfig  # noqa: E402
from ${package_name}.data import Features, Labels, make_synthetic_dataset  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    """Close every figure a test opened so the suite cannot leak memory."""
    yield
    plt.close("all")


@pytest.fixture
def config() -> ExperimentConfig:
    """A small, valid experiment configuration."""
    return ExperimentConfig(name="unit-test", seed=7, n_samples=120, n_classes=3)


@pytest.fixture
def dataset(config: ExperimentConfig) -> tuple[Features, Labels]:
    """Deterministic features and labels matching ``config``."""
    return make_synthetic_dataset(
        n_samples=config.n_samples,
        n_features=config.n_features,
        n_classes=config.n_classes,
        seed=config.seed,
    )


@pytest.fixture
def figures_dir(tmp_path: Path) -> Path:
    """An isolated directory for figures written during a test."""
    target = tmp_path / "figures"
    target.mkdir()
    return target

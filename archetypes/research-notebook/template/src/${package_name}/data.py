"""Reusable data helpers for ${project_name}.

These are the functions a notebook imports instead of redefining. Everything here
is deterministic given a seed, which is what makes a notebook's narrative
reproducible after you clear its outputs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray

Features = NDArray[np.float64]
Labels = NDArray[np.int64]


def rng_from_seed(seed: int) -> np.random.Generator:
    """Return a fresh, independent generator.

    Prefer this over ``np.random.seed``: global RNG state is exactly the kind of
    hidden dependency that makes a notebook unreproducible when cells run out of
    order.
    """
    if seed < 0:
        msg = f"seed must be non-negative, got {seed}"
        raise ValueError(msg)
    return np.random.default_rng(seed)


def make_synthetic_dataset(
    n_samples: int = 600,
    n_features: int = 2,
    n_classes: int = 3,
    seed: int = 42,
    cluster_std: float = 0.8,
) -> tuple[Features, Labels]:
    """Generate labelled Gaussian blobs — a dependency-free stand-in for real data.

    Lets the example notebook run end to end on a fresh clone with no download,
    so the template is verifiable. Swap this for your real loader.

    Returns:
        ``(features, labels)`` with shapes ``(n_samples, n_features)`` and
        ``(n_samples,)``.
    """
    if n_samples <= 0:
        msg = f"n_samples must be positive, got {n_samples}"
        raise ValueError(msg)
    if n_features < 2:
        msg = f"n_features must be at least 2, got {n_features}"
        raise ValueError(msg)
    if n_classes < 2:
        msg = f"n_classes must be at least 2, got {n_classes}"
        raise ValueError(msg)

    rng = rng_from_seed(seed)
    centers = rng.normal(loc=0.0, scale=4.0, size=(n_classes, n_features))
    labels = rng.integers(low=0, high=n_classes, size=n_samples).astype(np.int64)
    noise = rng.normal(loc=0.0, scale=cluster_std, size=(n_samples, n_features))
    features = (centers[labels] + noise).astype(np.float64)

    logger.debug(
        "Generated synthetic dataset: {} samples, {} features, {} classes (seed={})",
        n_samples,
        n_features,
        n_classes,
        seed,
    )
    return features, labels


def class_counts(labels: Labels) -> dict[int, int]:
    """Count samples per class, sorted by class id.

    A one-line sanity check every EDA notebook should run before anything else.
    """
    counts = Counter(int(label) for label in np.asarray(labels).ravel())
    return {label: counts[label] for label in sorted(counts)}


def class_balance(labels: Labels) -> float:
    """Return ``min_count / max_count`` — 1.0 is perfectly balanced, 0.0 degenerate."""
    counts = class_counts(labels)
    if not counts:
        return 0.0
    largest = max(counts.values())
    return min(counts.values()) / largest if largest else 0.0


def split_indices(
    n_samples: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Labels, Labels, Labels]:
    """Split ``range(n_samples)`` into disjoint train/val/test index arrays.

    The three arrays are guaranteed disjoint and to cover every index exactly
    once — the property that stops the same sample appearing in train and test.
    """
    if n_samples <= 0:
        msg = f"n_samples must be positive, got {n_samples}"
        raise ValueError(msg)
    if not 0.0 < train_fraction < 1.0:
        msg = f"train_fraction must be in (0, 1), got {train_fraction}"
        raise ValueError(msg)
    if not 0.0 < val_fraction < 1.0:
        msg = f"val_fraction must be in (0, 1), got {val_fraction}"
        raise ValueError(msg)
    if train_fraction + val_fraction >= 1.0:
        msg = (
            "train_fraction + val_fraction must be < 1.0 to leave a test split "
            f"(got {train_fraction} + {val_fraction})"
        )
        raise ValueError(msg)

    rng = rng_from_seed(seed)
    permutation = rng.permutation(n_samples).astype(np.int64)
    n_train = int(round(n_samples * train_fraction))
    n_val = int(round(n_samples * val_fraction))

    train = permutation[:n_train]
    val = permutation[n_train : n_train + n_val]
    test = permutation[n_train + n_val :]

    logger.debug("Split {} samples into {}/{}/{}", n_samples, train.size, val.size, test.size)
    return train, val, test


def save_array(array: Features | Labels, path: Path) -> Path:
    """Persist an array to ``path`` (creating parents) and return the path.

    Intermediate results belong in ``data/processed/``, not in a notebook's cell
    outputs — that is what keeps the notebook diff-able.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array))
    logger.info("Saved array with shape {} to {}", np.asarray(array).shape, path)
    return path

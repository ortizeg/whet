"""Tests for the reusable data helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ${package_name}.data import (
    Features,
    Labels,
    class_balance,
    class_counts,
    make_synthetic_dataset,
    rng_from_seed,
    save_array,
    split_indices,
)


def test_rng_is_deterministic() -> None:
    first = rng_from_seed(0).normal(size=5)
    second = rng_from_seed(0).normal(size=5)
    assert np.array_equal(first, second)


def test_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rng_from_seed(-1)


def test_dataset_shapes_and_dtypes(dataset: tuple[Features, Labels]) -> None:
    features, labels = dataset
    assert features.shape == (120, 2)
    assert labels.shape == (120,)
    assert features.dtype == np.float64
    assert labels.dtype == np.int64


def test_dataset_is_reproducible() -> None:
    a_features, a_labels = make_synthetic_dataset(n_samples=50, seed=3)
    b_features, b_labels = make_synthetic_dataset(n_samples=50, seed=3)
    assert np.array_equal(a_features, b_features)
    assert np.array_equal(a_labels, b_labels)


def test_different_seeds_give_different_data() -> None:
    a_features, _ = make_synthetic_dataset(n_samples=50, seed=3)
    b_features, _ = make_synthetic_dataset(n_samples=50, seed=4)
    assert not np.array_equal(a_features, b_features)


def test_labels_stay_in_range() -> None:
    _, labels = make_synthetic_dataset(n_samples=200, n_classes=4, seed=11)
    assert labels.min() >= 0
    assert labels.max() <= 3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_features": 1}, "n_features"),
        ({"n_classes": 1}, "n_classes"),
    ],
)
def test_dataset_validates_arguments(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_synthetic_dataset(**kwargs)


def test_class_counts_sums_to_dataset_size(dataset: tuple[Features, Labels]) -> None:
    _, labels = dataset
    counts = class_counts(labels)
    assert sum(counts.values()) == labels.size
    assert list(counts) == sorted(counts)


def test_class_balance_bounds() -> None:
    balanced = np.array([0, 0, 1, 1], dtype=np.int64)
    skewed = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    assert class_balance(balanced) == pytest.approx(1.0)
    assert class_balance(skewed) < 1.0


def test_splits_are_disjoint_and_complete() -> None:
    train, val, test = split_indices(100, 0.7, 0.15, seed=5)
    assert train.size + val.size + test.size == 100
    combined = np.concatenate([train, val, test])
    assert np.array_equal(np.sort(combined), np.arange(100))


def test_splits_are_reproducible() -> None:
    first = split_indices(100, seed=5)
    second = split_indices(100, seed=5)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


def test_splits_respect_requested_fractions() -> None:
    train, val, test = split_indices(1000, 0.6, 0.2, seed=1)
    assert train.size == 600
    assert val.size == 200
    assert test.size == 200


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 10, "train_fraction": 0.0}, "train_fraction"),
        ({"n_samples": 10, "val_fraction": 1.0}, "val_fraction"),
        ({"n_samples": 10, "train_fraction": 0.9, "val_fraction": 0.2}, "test split"),
    ],
)
def test_split_validates_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        split_indices(**kwargs)  # type: ignore[arg-type]


def test_save_array_roundtrip(tmp_path: Path) -> None:
    features, _ = make_synthetic_dataset(n_samples=20, seed=2)
    target = save_array(features, tmp_path / "nested" / "features.npy")
    assert target.is_file()
    assert np.array_equal(np.load(target), features)

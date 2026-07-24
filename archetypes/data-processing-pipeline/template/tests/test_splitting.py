"""The splitting tests — this is the file that protects the model from leakage."""

from __future__ import annotations

import polars as pl
import pytest

from ${package_name}.config import SplitConfig
from ${package_name}.splitting import (
    DataLeakageError,
    assert_no_group_leakage,
    group_aware_split,
    split_dataset,
    stratified_split,
)


def test_group_aware_split_produces_disjoint_groups(
    manifest: pl.DataFrame, split_config: SplitConfig
) -> None:
    """No group_id may appear in more than one split. This is the whole point."""
    splits = group_aware_split(manifest, split_config)

    groups = {name: set(frame["group_id"].to_list()) for name, frame in splits.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])


def test_group_aware_split_keeps_every_frame_of_a_clip_together(
    manifest: pl.DataFrame, split_config: SplitConfig
) -> None:
    splits = group_aware_split(manifest, split_config)
    home: dict[str, str] = {}
    for name, frame in splits.items():
        for group_id in frame["group_id"].to_list():
            assert home.setdefault(group_id, name) == name


def test_group_aware_split_preserves_all_records(
    manifest: pl.DataFrame, split_config: SplitConfig
) -> None:
    splits = group_aware_split(manifest, split_config)
    assert sum(frame.height for frame in splits.values()) == manifest.height
    recovered = {
        record_id for frame in splits.values() for record_id in frame["record_id"].to_list()
    }
    assert recovered == set(manifest["record_id"].to_list())


def test_split_is_deterministic_for_a_seed(
    manifest: pl.DataFrame, split_config: SplitConfig
) -> None:
    first = group_aware_split(manifest, split_config)
    second = group_aware_split(manifest, split_config)
    assert first["train"]["record_id"].to_list() == second["train"]["record_id"].to_list()


def test_different_seeds_give_different_splits(manifest: pl.DataFrame) -> None:
    a = group_aware_split(manifest, SplitConfig(seed=1))
    b = group_aware_split(manifest, SplitConfig(seed=2))
    assert a["test"]["group_id"].to_list() != b["test"]["group_id"].to_list()


def test_assert_no_group_leakage_detects_an_overlap(manifest: pl.DataFrame) -> None:
    leaky = {"train": manifest, "val": manifest.head(3), "test": manifest.tail(1)}
    with pytest.raises(DataLeakageError, match="data leakage"):
        assert_no_group_leakage(leaky, "group_id")


def test_random_row_split_would_leak(manifest: pl.DataFrame, split_config: SplitConfig) -> None:
    """The anti-pattern, demonstrated: a row-wise split leaks groups."""
    shuffled = manifest.sample(fraction=1.0, shuffle=True, seed=0)
    cut = int(shuffled.height * 0.7)
    row_split = {"train": shuffled[:cut], "test": shuffled[cut:]}
    with pytest.raises(DataLeakageError):
        assert_no_group_leakage(row_split, "group_id")

    # The group-aware split over the same manifest does not.
    assert_no_group_leakage(group_aware_split(manifest, split_config), "group_id")


def test_split_dataset_dispatches_to_stratified_without_a_group_column(
    manifest: pl.DataFrame,
) -> None:
    splits = split_dataset(manifest, SplitConfig(group_column=None))
    assert sum(frame.height for frame in splits.values()) == manifest.height


def test_stratified_split_covers_every_label(manifest: pl.DataFrame) -> None:
    splits = stratified_split(manifest, SplitConfig(group_column=None))
    for frame in splits.values():
        assert set(frame["label"].to_list()) == set(manifest["label"].to_list())


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)


def test_too_few_groups_is_an_error(tiny_manifest: pl.DataFrame, split_config: SplitConfig) -> None:
    with pytest.raises(ValueError, match="at least 3 distinct"):
        group_aware_split(tiny_manifest, split_config)


def test_missing_group_column_is_an_error(manifest: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="not in manifest"):
        group_aware_split(manifest.drop("group_id"), SplitConfig())

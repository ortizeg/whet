"""Leakage-free train/val/test splitting — the highest-risk step in the pipeline.

Correlated samples (frames from one clip, images of one patient, plays from one
match) are near-duplicates. If they straddle train/val/test, validation measures
memorization rather than generalization and the model fails silently in
production. So: **split on the group, never on the row**, whenever a grouping key
exists. Stratify by class only when no such key exists.

The disjointness check lives in the pipeline (``assert_no_group_leakage``), not
only in the test suite — a leak must abort the run, not fail a CI job later.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ${package_name}.config import SplitConfig

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
MIN_GROUPS = 3


class DataLeakageError(RuntimeError):
    """Raised when the same group appears in more than one split."""


def _deterministic_order(values: Sequence[str], seed: int) -> list[str]:
    """Shuffle deterministically by sorting on a seeded content hash.

    Reproducible across machines, Python versions and library upgrades — unlike
    ``random.shuffle``, whose stream is an implementation detail.
    """
    return sorted(values, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def _require_column(frame: pl.DataFrame, column: str) -> None:
    if column not in frame.columns:
        msg = f"Column {column!r} not in manifest (have: {frame.columns})"
        raise ValueError(msg)


def assert_no_group_leakage(splits: Mapping[str, pl.DataFrame], group_column: str) -> None:
    """Assert that no group appears in two splits. Raises ``DataLeakageError``.

    This is the invariant the whole pipeline exists to protect.
    """
    groups = {
        name: set(frame[group_column].unique().to_list())
        for name, frame in splits.items()
        if frame.height > 0
    }
    names = sorted(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                sample = sorted(overlap)[:5]
                msg = (
                    f"{left}/{right} share {len(overlap)} group(s) on {group_column!r} "
                    f"— data leakage. Examples: {sample}"
                )
                raise DataLeakageError(msg)
    logger.info("Leakage check passed: {} splits disjoint on {!r}", len(groups), group_column)


def _log_realized(splits: Mapping[str, pl.DataFrame], config: SplitConfig) -> None:
    """Group-aware splits skew ratios when group sizes are uneven — always look."""
    total = sum(frame.height for frame in splits.values())
    for name in SPLIT_NAMES:
        frame = splits[name]
        fraction = frame.height / total if total else 0.0
        groups = frame[config.group_column].n_unique() if config.group_column else 0
        logger.info(
            "Split '{}': {} records ({:.1%}), {} groups, {} labels",
            name,
            frame.height,
            fraction,
            groups,
            frame[config.stratify_column].n_unique(),
        )


def _reject_empty(splits: Mapping[str, pl.DataFrame]) -> None:
    empty = [name for name, frame in splits.items() if frame.height == 0]
    if empty:
        msg = f"Splits {empty} are empty — adjust ratios or collect more data"
        raise ValueError(msg)


def group_aware_split(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Assign whole groups to a split so no group straddles train/val/test."""
    key = config.group_column
    if key is None:
        msg = "group_aware_split requires SplitConfig.group_column to be set"
        raise ValueError(msg)
    _require_column(manifest, key)
    _require_column(manifest, config.stratify_column)

    groups = _deterministic_order(manifest[key].unique().to_list(), config.seed)
    n_groups = len(groups)
    if n_groups < MIN_GROUPS:
        msg = f"Need at least {MIN_GROUPS} distinct {key!r} values to split, got {n_groups}"
        raise ValueError(msg)

    n_train = min(max(int(n_groups * config.train_ratio), 1), n_groups - 2)
    n_val = min(max(int(n_groups * config.val_ratio), 1), n_groups - n_train - 1)
    assignment = {
        "train": groups[:n_train],
        "val": groups[n_train : n_train + n_val],
        "test": groups[n_train + n_val :],
    }

    splits = {
        name: manifest.filter(pl.col(key).is_in(members)) for name, members in assignment.items()
    }
    assert_no_group_leakage(splits, key)
    _reject_empty(splits)
    _log_realized(splits, config)
    return splits


def stratified_split(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Per-class random split — only valid when there is no grouping key."""
    column = config.stratify_column
    _require_column(manifest, column)
    _require_column(manifest, "record_id")

    parts: dict[str, list[pl.DataFrame]] = {name: [] for name in SPLIT_NAMES}
    for label in sorted(manifest[column].unique().to_list()):
        rows = manifest.filter(pl.col(column) == label)
        order = _deterministic_order(rows["record_id"].to_list(), config.seed)
        cut_train = int(len(order) * config.train_ratio)
        cut_val = cut_train + int(len(order) * config.val_ratio)
        assignment = {
            "train": order[:cut_train],
            "val": order[cut_train:cut_val],
            "test": order[cut_val:],
        }
        for name, members in assignment.items():
            parts[name].append(rows.filter(pl.col("record_id").is_in(members)))

    splits = {name: pl.concat(chunks) for name, chunks in parts.items()}
    _reject_empty(splits)
    _log_realized(splits, config)
    return splits


def split_dataset(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Group-aware split when ``group_column`` is set, stratified otherwise."""
    if config.group_column is not None:
        logger.info("Splitting group-aware on {!r} (seed {})", config.group_column, config.seed)
        return group_aware_split(manifest, config)
    logger.warning(
        "No group_column set — falling back to a stratified split on {!r}. "
        "This is only safe for genuinely independent samples.",
        config.stratify_column,
    )
    return stratified_split(manifest, config)

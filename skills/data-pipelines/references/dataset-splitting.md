# Dataset Splitting Without Leakage

Group-aware and stratified splitting for CV/ML datasets — the highest-risk step in any pipeline.

## Contents

- [Why Group-Aware Splitting Matters](#why-group-aware-splitting-matters)
- [Implementation](#implementation)
- [The Anti-Pattern](#the-anti-pattern)
- [Skewed Ratios](#skewed-ratios)

## Why Group-Aware Splitting Matters

**This is the highest-risk step in any CV pipeline.** Correlated samples — frames
from the same video, images of the same patient, plays from the same match, crops
of the same player — are near-duplicates. If they straddle train/val/test,
validation measures memorization rather than generalization and the model fails
silently in production. Split on the *group*, never on the row, whenever a grouping
key exists: video/clip ID, patient/subject ID, match/session ID, player identity,
camera or site ID, or capture date for time-correlated data. Stratify only when no
such key exists.

## Implementation

```python
"""Dataset splitting with stratification and group-aware leak prevention."""

from __future__ import annotations  # for the SplitConfig self-reference

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field, model_validator


class SplitConfig(BaseModel):
    """Configuration for dataset splitting."""

    train_ratio: float = Field(gt=0.0, lt=1.0, default=0.7)
    val_ratio: float = Field(gt=0.0, lt=1.0, default=0.15)
    test_ratio: float = Field(gt=0.0, lt=1.0, default=0.15)
    seed: int = 42
    stratify_column: str = "label"
    group_column: str | None = None  # set whenever a grouping key exists

    @model_validator(mode="after")
    def ratios_must_sum_to_one(self) -> SplitConfig:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            msg = f"Split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


def split_dataset(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Group-aware split when group_column is set, otherwise stratified."""
    if config.group_column is not None:
        return group_aware_split(manifest, config)
    return stratified_split(manifest, config)


def group_aware_split(df: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Assign whole groups to a split so no group straddles train/val/test."""
    key = config.group_column
    assert key is not None
    groups = df[key].unique().sample(fraction=1.0, seed=config.seed, shuffle=True)
    n_train = int(len(groups) * config.train_ratio)
    n_val = int(len(groups) * config.val_ratio)

    train_groups = set(groups[:n_train].to_list())
    val_groups = set(groups[n_train : n_train + n_val].to_list())
    test_groups = set(groups[n_train + n_val :].to_list())

    # Leakage checks belong in the pipeline, not only in tests.
    pairs = {"train/val": (train_groups, val_groups), "train/test": (train_groups, test_groups)}
    pairs["val/test"] = (val_groups, test_groups)
    for name, (a, b) in pairs.items():
        if not a.isdisjoint(b):
            msg = f"{name} group overlap — data leakage"
            raise ValueError(msg)

    splits = {
        "train": df.filter(pl.col(key).is_in(train_groups)),
        "val": df.filter(pl.col(key).is_in(val_groups)),
        "test": df.filter(pl.col(key).is_in(test_groups)),
    }
    for name, part in splits.items():
        logger.info("Split '{}': {} records ({} groups)", name, len(part), part[key].n_unique())
    return splits


def stratified_split(df: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Per-class stratified random split — only when there is no grouping key."""
    parts: dict[str, list[pl.DataFrame]] = {"train": [], "val": [], "test": []}
    for label in sorted(df[config.stratify_column].unique().to_list()):
        rows = df.filter(pl.col(config.stratify_column) == label).sample(
            fraction=1.0, seed=config.seed, shuffle=True
        )
        a = int(len(rows) * config.train_ratio)
        b = a + int(len(rows) * config.val_ratio)
        parts["train"].append(rows[:a])
        parts["val"].append(rows[a:b])
        parts["test"].append(rows[b:])
    return {name: pl.concat(chunks) for name, chunks in parts.items()}
```

## The Anti-Pattern

A plain random split silently leaks every group:

```python
# WRONG: neighbouring frames of the same clip land in both train and test.
train, val, test = np.split(df.sample(frac=1), [int(0.7 * len(df)), int(0.85 * len(df))])
```

## Skewed Ratios

Group-aware splits skew ratios when group sizes are uneven. Always log realized
per-split record counts and class distribution and check them before training;
rebalance by greedily assigning the largest groups first if needed.

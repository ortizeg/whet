"""Concrete ETL stages: ``ingest -> validate -> split -> write``.

Stages are plain functions with explicit inputs and outputs — no framework, no
registry, no abstract base class. Add a stage by writing another function and
calling it from ``pipeline.run_pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from ${package_name}.manifest import (
    build_manifest,
    make_record,
    write_dataset_card,
    write_manifest,
)
from ${package_name}.quality import validate_manifest, validate_splits
from ${package_name}.splitting import assert_no_group_leakage, split_dataset

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ${package_name}.config import (
        IngestConfig,
        OutputConfig,
        QualityConfig,
        SplitConfig,
    )
    from ${package_name}.manifest import ImageRecord
    from ${package_name}.quality import QualityReport


def stage_ingest(config: IngestConfig) -> pl.DataFrame:
    """Scan the raw tree and build a content-hashed manifest.

    Expects ``<root>/<label>/<group_id>/<file>``; anything shallower or with an
    unexpected extension is skipped and counted, never silently dropped.
    """
    root = config.root
    if not root.is_dir():
        msg = f"Ingest root {root} does not exist"
        raise FileNotFoundError(msg)

    needed = max(config.label_depth, config.group_depth) + 1
    records: list[ImageRecord] = []
    skipped_extension = 0
    skipped_layout = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in config.extensions:
            skipped_extension += 1
            continue
        parts = path.relative_to(root).parts[:-1]
        if len(parts) < needed:
            skipped_layout += 1
            logger.warning("Skipping {}: expected at least {} parent directories", path, needed)
            continue
        records.append(
            make_record(
                path=path,
                label=parts[config.label_depth],
                group_id=parts[config.group_depth],
                root=root,
            )
        )

    logger.info(
        "Ingest: {} record(s) from {} — skipped {} (extension), {} (layout)",
        len(records),
        root,
        skipped_extension,
        skipped_layout,
    )
    return build_manifest(records)


def stage_validate(manifest: pl.DataFrame, config: QualityConfig) -> QualityReport:
    """Gate the manifest on data quality. Errors abort the run."""
    report = validate_manifest(manifest, config)
    report.raise_for_errors()
    return report


def stage_split(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Split into train/val/test, group-aware unless told otherwise."""
    splits = split_dataset(manifest, config)
    if config.group_column is not None:
        assert_no_group_leakage(splits, config.group_column)
    return splits


def stage_write(
    manifest: pl.DataFrame,
    splits: Mapping[str, pl.DataFrame],
    config: OutputConfig,
    quality: QualityConfig,
) -> dict[str, Path]:
    """Validate the splits, then write Parquet artifacts and the dataset card."""
    validate_splits(splits, quality).raise_for_errors()

    out = config.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {
        "manifest": write_manifest(manifest, out / "manifest.parquet", config.compression)
    }
    for name, frame in sorted(splits.items()):
        artifacts[name] = write_manifest(
            frame.with_columns(pl.lit(name).alias("split")),
            out / "splits" / f"{name}.parquet",
            config.compression,
        )
    artifacts["card"] = write_dataset_card(
        out / "dataset.json",
        manifest,
        [(name, frame.height) for name, frame in sorted(splits.items())],
    )
    return artifacts

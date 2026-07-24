"""Shared fixtures: synthetic manifests and a synthetic raw tree on disk."""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from ${package_name}.config import (
    IngestConfig,
    OutputConfig,
    PipelineConfig,
    QualityConfig,
    SplitConfig,
)
from ${package_name}.manifest import ImageRecord, build_manifest
from ${package_name}.sample_data import generate_sample_dataset

LABELS: tuple[str, ...] = ("goal", "no_goal")


def make_records(
    labels: tuple[str, ...] = LABELS,
    groups_per_label: int = 5,
    frames_per_group: int = 4,
) -> list[ImageRecord]:
    """Build correlated records: many frames per clip, many clips per label."""
    records: list[ImageRecord] = []
    for label in labels:
        for group_index in range(groups_per_label):
            group_id = f"{label}_clip_{group_index:03d}"
            for frame_index in range(frames_per_group):
                relative = f"{label}/{group_id}/frame_{frame_index:04d}.png"
                content = hashlib.sha256(relative.encode()).hexdigest()
                records.append(
                    ImageRecord(
                        record_id=content[:16],
                        path=relative,
                        label=label,
                        group_id=group_id,
                        content_sha256=content,
                        size_bytes=128 + frame_index,
                    )
                )
    return records


@pytest.fixture
def records() -> list[ImageRecord]:
    return make_records()


@pytest.fixture
def manifest(records: list[ImageRecord]) -> pl.DataFrame:
    return build_manifest(records)


@pytest.fixture
def tiny_manifest() -> pl.DataFrame:
    """Too few groups to split — used to prove the splitter refuses rather than leaks."""
    return build_manifest(make_records(labels=("goal",), groups_per_label=2, frames_per_group=2))


@pytest.fixture
def split_config() -> SplitConfig:
    return SplitConfig()


@pytest.fixture
def quality_config() -> QualityConfig:
    return QualityConfig()


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    generate_sample_dataset(root, labels=LABELS, groups_per_label=4, frames_per_group=5)
    return root


@pytest.fixture
def pipeline_config(raw_root: Path, tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(
        name="test-pipeline",
        ingest=IngestConfig(root=raw_root),
        quality=QualityConfig(min_groups_per_label=2),
        split=SplitConfig(),
        output=OutputConfig(processed_dir=tmp_path / "processed"),
    )

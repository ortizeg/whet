"""Data-quality gate behaviour."""

from __future__ import annotations

import polars as pl
import pytest

from ${package_name}.config import QualityConfig, SplitConfig
from ${package_name}.manifest import build_manifest
from ${package_name}.quality import Severity, validate_manifest, validate_splits
from ${package_name}.splitting import group_aware_split


def test_a_clean_manifest_passes(manifest: pl.DataFrame, quality_config: QualityConfig) -> None:
    report = validate_manifest(manifest, quality_config)
    assert report.ok
    assert report.n_records == manifest.height


def test_empty_manifest_is_an_error(quality_config: QualityConfig) -> None:
    report = validate_manifest(build_manifest([]), quality_config)
    assert not report.ok
    assert report.errors[0].check == "non_empty"


def test_missing_columns_are_reported(
    manifest: pl.DataFrame, quality_config: QualityConfig
) -> None:
    report = validate_manifest(manifest.drop("content_sha256"), quality_config)
    assert not report.ok
    assert report.errors[0].check == "schema"


def test_duplicate_content_is_an_error(
    manifest: pl.DataFrame, quality_config: QualityConfig
) -> None:
    duplicated = pl.concat([manifest, manifest.head(2)])
    report = validate_manifest(duplicated, quality_config)
    assert not report.ok
    assert {issue.check for issue in report.errors} == {"duplicate_content"}


def test_duplicate_content_can_be_downgraded(manifest: pl.DataFrame) -> None:
    duplicated = pl.concat([manifest, manifest.head(2)])
    report = validate_manifest(duplicated, QualityConfig(allow_duplicate_content=True))
    assert report.ok
    assert any(issue.severity is Severity.WARNING for issue in report.issues)


def test_undersized_files_are_an_error(manifest: pl.DataFrame) -> None:
    report = validate_manifest(manifest, QualityConfig(min_file_bytes=10_000))
    assert not report.ok
    assert report.errors[0].check == "file_size"


def test_thin_labels_are_an_error(manifest: pl.DataFrame) -> None:
    report = validate_manifest(manifest, QualityConfig(min_groups_per_label=99))
    assert not report.ok
    assert "label_group_support" in {issue.check for issue in report.errors}


def test_raise_for_errors_aborts(manifest: pl.DataFrame) -> None:
    report = validate_manifest(manifest, QualityConfig(min_file_bytes=10_000))
    with pytest.raises(ValueError, match="Data quality gate failed"):
        report.raise_for_errors()


def test_healthy_splits_pass(manifest: pl.DataFrame, quality_config: QualityConfig) -> None:
    splits = group_aware_split(manifest, SplitConfig())
    report = validate_splits(splits, quality_config)
    assert report.ok
    assert report.n_records == manifest.height


def test_split_validation_flags_a_missing_label(
    manifest: pl.DataFrame, quality_config: QualityConfig
) -> None:
    first_label = manifest["label"][0]
    splits = {
        "train": manifest,
        "val": manifest.filter(pl.col("label") == first_label),
        "test": manifest,
    }
    report = validate_splits(splits, quality_config)
    assert "split_label_coverage" in {issue.check for issue in report.issues}


def test_split_validation_flags_an_empty_split(
    manifest: pl.DataFrame, quality_config: QualityConfig
) -> None:
    splits = {"train": manifest, "val": manifest.head(0), "test": manifest}
    report = validate_splits(splits, quality_config)
    assert not report.ok
    assert report.errors[0].check == "split_non_empty"

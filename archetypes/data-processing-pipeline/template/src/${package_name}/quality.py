"""Data-quality gate — fail fast, fail loud, before anything downstream runs.

No public or "curated" dataset is clean by default. Validation runs after
extraction and again after splitting: a healthy manifest still yields empty or
single-class splits when the ratios are wrong.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

from ${package_name}.manifest import MANIFEST_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ${package_name}.config import QualityConfig


class Severity(StrEnum):
    """How badly a check failed."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QualityIssue(BaseModel, frozen=True):
    """A single finding from one check."""

    check: str
    severity: Severity
    message: str
    count: int = Field(default=0, ge=0)


class QualityReport(BaseModel, frozen=True):
    """The outcome of a validation pass."""

    n_records: int = Field(ge=0)
    issues: tuple[QualityIssue, ...] = ()

    @property
    def errors(self) -> tuple[QualityIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        """Abort the run if any check failed at ERROR severity."""
        if self.errors:
            detail = "; ".join(f"[{issue.check}] {issue.message}" for issue in self.errors)
            msg = f"Data quality gate failed with {len(self.errors)} error(s): {detail}"
            raise ValueError(msg)

    def log(self) -> None:
        for issue in self.issues:
            log = {
                Severity.ERROR: logger.error,
                Severity.WARNING: logger.warning,
                Severity.INFO: logger.info,
            }[issue.severity]
            log("[{}] {} (n={})", issue.check, issue.message, issue.count)
        logger.info(
            "Quality report: {} records, {} error(s), {} issue(s) total",
            self.n_records,
            len(self.errors),
            len(self.issues),
        )


def validate_manifest(manifest: pl.DataFrame, config: QualityConfig) -> QualityReport:
    """Run every quality check over a manifest and collect the findings."""
    issues: list[QualityIssue] = []

    missing = [column for column in MANIFEST_SCHEMA if column not in manifest.columns]
    if missing:
        issues.append(
            QualityIssue(
                check="schema",
                severity=Severity.ERROR,
                message=f"Manifest missing columns: {missing}",
                count=len(missing),
            )
        )
        return QualityReport(n_records=manifest.height, issues=tuple(issues))

    if manifest.height == 0:
        issues.append(
            QualityIssue(check="non_empty", severity=Severity.ERROR, message="Manifest is empty")
        )
        return QualityReport(n_records=0, issues=tuple(issues))

    issues.extend(_check_file_sizes(manifest, config))
    issues.extend(_check_duplicates(manifest, config))
    issues.extend(_check_label_support(manifest, config))
    issues.extend(_check_group_coverage(manifest))

    report = QualityReport(n_records=manifest.height, issues=tuple(issues))
    report.log()
    return report


def _check_file_sizes(manifest: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    undersized = manifest.filter(pl.col("size_bytes") < config.min_file_bytes)
    if undersized.height == 0:
        return []
    return [
        QualityIssue(
            check="file_size",
            severity=Severity.ERROR,
            message=(
                f"{undersized.height} file(s) smaller than {config.min_file_bytes} bytes, "
                f"e.g. {undersized['path'].to_list()[:3]}"
            ),
            count=undersized.height,
        )
    ]


def _check_duplicates(manifest: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    duplicated = manifest.height - manifest["content_sha256"].n_unique()
    if duplicated == 0:
        return []
    severity = Severity.WARNING if config.allow_duplicate_content else Severity.ERROR
    return [
        QualityIssue(
            check="duplicate_content",
            severity=severity,
            message=f"{duplicated} record(s) share a content hash with another record",
            count=duplicated,
        )
    ]


def _check_label_support(manifest: pl.DataFrame, config: QualityConfig) -> list[QualityIssue]:
    counts = manifest.group_by("label").agg(
        pl.len().alias("n_records"),
        pl.col("group_id").n_unique().alias("n_groups"),
    )
    issues: list[QualityIssue] = []
    for row in counts.iter_rows(named=True):
        if row["n_records"] < config.min_records_per_label:
            issues.append(
                QualityIssue(
                    check="label_support",
                    severity=Severity.ERROR,
                    message=(
                        f"Label {row['label']!r} has {row['n_records']} record(s), "
                        f"minimum is {config.min_records_per_label}"
                    ),
                    count=row["n_records"],
                )
            )
        if row["n_groups"] < config.min_groups_per_label:
            issues.append(
                QualityIssue(
                    check="label_group_support",
                    severity=Severity.ERROR,
                    message=(
                        f"Label {row['label']!r} spans {row['n_groups']} group(s), "
                        f"minimum is {config.min_groups_per_label}"
                    ),
                    count=row["n_groups"],
                )
            )
    return issues


def _check_group_coverage(manifest: pl.DataFrame) -> list[QualityIssue]:
    """A group that carries two labels is usually an annotation bug worth surfacing."""
    mixed = (
        manifest.group_by("group_id")
        .agg(pl.col("label").n_unique().alias("n_labels"))
        .filter(pl.col("n_labels") > 1)
    )
    if mixed.height == 0:
        return []
    return [
        QualityIssue(
            check="mixed_label_group",
            severity=Severity.WARNING,
            message=(
                f"{mixed.height} group(s) carry more than one label, "
                f"e.g. {mixed['group_id'].to_list()[:3]}"
            ),
            count=mixed.height,
        )
    ]


def validate_splits(splits: Mapping[str, pl.DataFrame], config: QualityConfig) -> QualityReport:
    """Post-split gate: every split must be non-empty and cover every label."""
    issues: list[QualityIssue] = []
    all_labels = {label for frame in splits.values() for label in frame["label"].unique().to_list()}
    total = sum(frame.height for frame in splits.values())
    for name, frame in sorted(splits.items()):
        if frame.height == 0:
            issues.append(
                QualityIssue(
                    check="split_non_empty",
                    severity=Severity.ERROR,
                    message=f"Split {name!r} is empty",
                )
            )
            continue
        missing = sorted(all_labels - set(frame["label"].unique().to_list()))
        if missing:
            issues.append(
                QualityIssue(
                    check="split_label_coverage",
                    severity=Severity.WARNING,
                    message=f"Split {name!r} is missing label(s) {missing}",
                    count=len(missing),
                )
            )
        if frame.height < config.min_records_per_label:
            issues.append(
                QualityIssue(
                    check="split_size",
                    severity=Severity.WARNING,
                    message=f"Split {name!r} has only {frame.height} record(s)",
                    count=frame.height,
                )
            )
    report = QualityReport(n_records=total, issues=tuple(issues))
    report.log()
    return report

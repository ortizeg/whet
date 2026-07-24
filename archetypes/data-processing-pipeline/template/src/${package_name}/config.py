"""Pydantic V2 configuration models for the ${project_name} pipeline.

Every stage is configured by a frozen model so that a bad parameter fails at load
time rather than three hours into a run. Paths are interpreted relative to the
working directory the pipeline is invoked from.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")


class IngestConfig(BaseModel, frozen=True):
    """Where raw data lives and how records are derived from it.

    The expected on-disk layout is ``<root>/<label>/<group_id>/<file>`` — for
    example ``data/raw/goal/match_0007/frame_000123.png``. The middle directory
    is the *grouping key*: every frame of one clip, every image of one patient,
    every crop of one player. It is what the splitter keeps together.
    """

    root: Path = Path("data/raw")
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    label_depth: int = Field(default=0, ge=0)
    group_depth: int = Field(default=1, ge=0)

    @field_validator("extensions")
    @classmethod
    def _normalize_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            msg = "extensions must not be empty"
            raise ValueError(msg)
        return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value)

    @model_validator(mode="after")
    def _depths_must_differ(self) -> Self:
        if self.label_depth == self.group_depth:
            msg = "label_depth and group_depth must refer to different path components"
            raise ValueError(msg)
        return self


class QualityConfig(BaseModel, frozen=True):
    """Thresholds for the data-quality gate that runs after ingestion."""

    min_file_bytes: int = Field(default=1, ge=0)
    min_records_per_label: int = Field(default=1, ge=0)
    min_groups_per_label: int = Field(default=1, ge=0)
    allow_duplicate_content: bool = False


class SplitConfig(BaseModel, frozen=True):
    """Train/val/test ratios and the columns that drive the split.

    ``group_column`` is the leakage guard. Leave it set whenever a grouping key
    exists; set it to ``None`` only for genuinely independent samples, in which
    case a stratified split on ``stratify_column`` is used instead.
    """

    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    seed: int = 42
    group_column: str | None = "group_id"
    stratify_column: str = "label"

    @model_validator(mode="after")
    def _ratios_must_sum_to_one(self) -> Self:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            msg = f"Split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)
        return self


Compression = Literal["zstd", "snappy", "gzip", "lz4", "uncompressed"]


class OutputConfig(BaseModel, frozen=True):
    """Where processed artifacts are written."""

    processed_dir: Path = Path("data/processed")
    compression: Compression = "zstd"


class PipelineConfig(BaseModel, frozen=True):
    """Top-level pipeline configuration."""

    name: str = "${project_slug}"
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_toml(cls, path: Path) -> PipelineConfig:
        """Load and validate a pipeline configuration from a TOML file."""
        with path.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
        return cls.model_validate(raw)

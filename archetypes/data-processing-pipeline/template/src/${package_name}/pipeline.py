"""Pipeline orchestration — run the stages in order and report what happened."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from ${package_name}.config import PipelineConfig
from ${package_name}.manifest import manifest_fingerprint
from ${package_name}.stages import stage_ingest, stage_split, stage_validate, stage_write


class PipelineResult(BaseModel, frozen=True):
    """Everything a run produced — enough to trace data lineage after the fact."""

    name: str
    fingerprint: str
    n_records: int = Field(ge=0)
    n_groups: int = Field(ge=0)
    split_counts: dict[str, int] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run ``ingest -> validate -> split -> write`` end to end."""
    logger.info("Pipeline {!r} starting from {}", config.name, config.ingest.root)

    manifest = stage_ingest(config.ingest)
    stage_validate(manifest, config.quality)
    splits = stage_split(manifest, config.split)
    artifacts = stage_write(manifest, splits, config.output, config.quality)

    result = PipelineResult(
        name=config.name,
        fingerprint=manifest_fingerprint(manifest),
        n_records=manifest.height,
        n_groups=manifest["group_id"].n_unique(),
        split_counts={name: frame.height for name, frame in sorted(splits.items())},
        artifacts=artifacts,
    )
    logger.success(
        "Pipeline {!r} finished: {} records, {} groups, fingerprint {}",
        result.name,
        result.n_records,
        result.n_groups,
        result.fingerprint[:12],
    )
    return result

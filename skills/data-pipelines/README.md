# Data Pipelines Skill

## Purpose

This skill covers the design of dataset and ETL/ELT pipelines for computer vision
and machine learning projects: building validated dataset manifests, splitting data
without leakage, selecting a storage format, validating data quality, designing
augmentation pipelines, processing data at scale, and evolving dataset schemas over
time. The purpose is to make the data layer reproducible and trustworthy, so that
training metrics reflect generalization rather than accidental memorization.

The highest-value part of the skill is group-aware splitting: frames from the same
video, images of the same patient, or plays from the same match must never straddle
train, val, and test.

## When to Use

Reference this skill when:

- Designing a data ingestion or ETL pipeline for image, video, or tabular ML data.
- Splitting a dataset into train/val/test, especially when samples are correlated.
- Diagnosing suspiciously high validation scores that may come from data leakage.
- Choosing a storage format: Parquet, LMDB, WebDataset, TFRecord, Arrow, or raw files.
- Writing data quality checks or Great Expectations suites for a dataset.
- Designing an augmentation pipeline that stays deterministic on val and test.
- Processing datasets that exceed memory (chunked parallel work, streaming reads).
- Changing the fields of an existing dataset and needing a migration path.

Use `model-evaluation` for scoring models on the splits this skill produces.

## Key Patterns

- **Validated manifest** — one Pydantic-checked row per sample, with a SHA-256
  content hash, written as compressed Parquet.
- **Group-aware splitting** — assign whole groups to a split and assert disjointness
  in the pipeline, not only in tests.
- **Stratified splitting** — per-class random split, used only when no grouping key
  exists.
- **Storage decision matrix** — Parquet for tabular, LMDB for single-node random
  access, WebDataset for distributed/object-storage training, raw files while small.
- **Two-layer validation** — Pydantic per record, Great Expectations in aggregate,
  plus a cheap in-pipeline gate run after extraction and after splitting.
- **Config-driven augmentation** — a single `strength` knob scales training
  augmentations; val/test transforms stay deterministic.
- **Chunked parallel processing** — per-item error isolation, progress logging, and
  lazy `scan_parquet` slices for data larger than RAM.
- **Versioned schemas** — one model per version, an explicit migration per hop, and
  `schema_version` stored alongside the data.

See `SKILL.md` for complete documentation and code examples.

---
name: data-pipelines
description: >
  Use this skill when designing a dataset or ETL/ELT pipeline, choosing a storage format
  (Parquet, LMDB, WebDataset, TFRecord), splitting a dataset into train/val/test, preventing
  data leakage across groups (patient, video, match, player, scene), validating data quality,
  evolving a dataset schema, or processing image/video data at a scale that exceeds memory.
  Reach for it any time you would otherwise glob a directory, random-split a DataFrame, or
  hand-write a CSV manifest, even if the user doesn't say "pipeline" or "ETL" explicitly.
  Not for scoring or comparing models (see `model-evaluation`).
---

# Data Pipelines for CV/ML

Reproducible, scalable pipelines that feed ML training and inference. This page is the
index: core principles, the leakage rule, and the pipeline shape are inline; the full
patterns live in `references/` and should be read only when the task calls for them.

## Core Principles

1. **Data quality first.** No model compensates for bad data. Validate early and often.
2. **Reproducibility.** Every version, transformation, and split is traceable via content hashes and seeded, deterministic code.
3. **Schema as contract.** Schemas bind pipeline stages together. Changes require explicit migrations, never silent mutations.
4. **Immutable datasets.** Published versions are never modified in place; new versions carry lineage to their source.
5. **Fail fast, fail loud.** Raise on quality violations immediately rather than propagating corrupt data downstream.

## Pipeline Shape

A pipeline is a fixed sequence of named stages — `extract → validate_raw → transform →
validate_transformed → load` — with a real gate between each. The manifest is the source of
truth: one validated row per sample with a content hash, written as Parquet (never CSV).
`split` is deliberately *not* a manifest field — it is assigned downstream so a dataset can
be re-split without re-extracting.

```
Building a data pipeline?
├── Small (< 10 GB) ............ Polars / Pandas
├── Medium (10–500 GB) ......... Polars / DuckDB + Parquet
├── Large (500 GB – 10 TB) ..... Arrow / Spark + cloud object storage
├── Streaming / continuous ..... Kafka / Flink + Delta Lake
└── Annotation pipeline ........ Label Studio / CVAT + validation hooks
```

## Splitting Without Leakage — The Single Highest-Value Rule

**This is the highest-risk step in any CV pipeline.** Correlated samples — frames from the
same video, images of the same patient, plays from the same match, crops of the same player
— are near-duplicates. If they straddle train/val/test, validation measures memorization
rather than generalization and the model fails silently in production. **Split on the
*group*, never on the row**, whenever a grouping key exists: video/clip ID, patient/subject
ID, match/session ID, player identity, camera or site ID, or capture date for
time-correlated data. Stratify by class only when no such key exists.

```python
def split_dataset(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]:
    """Group-aware split when group_column is set, otherwise stratified."""
    if config.group_column is not None:
        return group_aware_split(manifest, config)
    return stratified_split(manifest, config)


# Inside group_aware_split: leakage checks belong in the pipeline, not only in tests.
for name, (a, b) in pairs.items():  # train/val, train/test, val/test
    if not a.isdisjoint(b):
        msg = f"{name} group overlap — data leakage"
        raise ValueError(msg)
```

```python
# WRONG: neighbouring frames of the same clip land in both train and test.
train, val, test = np.split(df.sample(frac=1), [int(0.7 * len(df)), int(0.85 * len(df))])
```

Group-aware splits skew ratios when group sizes are uneven — always log realized per-split
record counts and class distribution and check them before training. Full implementation:
`references/dataset-splitting.md`.

## Conventions

- Configure every stage with a validated **Pydantic** model; bad parameters must fail at load time, not three hours in.
- **Parquet** (`compression="zstd"`) for all tabular data; LMDB for single-node random access; WebDataset for multi-node streaming.
- **Content-hash** every sample (SHA-256) for deduplication and corruption detection.
- **Log with loguru** at every stage boundary — realized counts, not just "done".
- Validate **after extraction and after splitting**; a healthy manifest still yields empty or single-class splits when ratios are wrong.
- Keep augmentation **config-driven** with one `strength` knob, and log the config with the run.
- Version schemas explicitly and store `schema_version` alongside the data.

## Anti-Patterns

- **Never split randomly when a grouping key exists.** Frames from one video, images of one patient, or plays from one match must live in exactly one split.
- **Never split without stratification** when no grouping key exists — class imbalance across splits makes metrics unreliable.
- **Never modify raw data in place.** Write transformations to a separate directory; keep originals intact.
- **Never skip validation between pipeline stages.** Catching corrupt data early saves hours of wasted training, and no public or "curated" dataset is clean by default.
- **Never use CSV for large datasets.** Parquet for tabular, LMDB for random-access images, WebDataset for streaming.
- **Never apply training augmentations to val or test.** Only deterministic resize/normalize outside training.
- **Never silently change a schema.** Version it and write an explicit migration.
- **Never hard-code paths or ratios.** Put them in a validated Pydantic config.
- **Never version data in Git.** Keep large artifacts in object storage and track a manifest.

## Deep dives

- `references/dataset-splitting.md` — read when splitting a dataset, choosing a grouping key, or auditing an existing split for leakage. Full `SplitConfig`, `group_aware_split`, and `stratified_split`.
- `references/pipeline-architecture.md` — read when standing up a new pipeline: sizing the stack, the Pydantic `PipelineConfig`, and building the hashed Parquet manifest.
- `references/storage-formats.md` — read when choosing between Parquet, LMDB, WebDataset, TFRecord, Arrow, or raw folders, or packing images into LMDB.
- `references/data-quality.md` — read when adding validation: Great Expectations suites for a manifest and the cheap in-pipeline quality gate.
- `references/augmentation.md` — read when designing an augmentation pipeline or wiring Albumentations to a config with a single strength knob.
- `references/large-scale-processing.md` — read when data exceeds RAM or a single core: chunked `ProcessPoolExecutor` processing and lazy Polars scans.
- `references/schema-evolution.md` — read when a dataset schema must change: versioned Pydantic models, the migration registry, and when not to subclass.

## Related Skills

- `pydantic` — validated configs and schema definitions for every stage
- `pytorch-lightning` — `LightningDataModule` consuming validated splits
- `model-evaluation` — scoring models on the splits this skill produces
- `testing` — unit tests for transforms, integration tests for pipeline stages

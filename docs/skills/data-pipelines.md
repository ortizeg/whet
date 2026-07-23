# Data Pipelines

The data-pipelines skill covers dataset and ETL/ELT design for CV/ML projects: validated manifests, leakage-free splitting, storage format selection, data quality validation, augmentation design, large-scale processing, and schema evolution.

**Skill directory:** `skills/data-pipelines/`

## Purpose

Model quality is bounded by data quality. Most silent failures in computer vision projects are data failures: near-duplicate frames leaking across splits, corrupt images that never raised an error, a schema that gained a column without anyone recording it, or a CSV manifest that became the bottleneck at 10 million rows. This skill teaches Claude Code to build data layers that are reproducible and self-checking -- content-hashed manifests, group-aware splits, validation gates between pipeline stages, and versioned schemas with explicit migrations.

## When to Use

- Designing an ingestion or ETL pipeline for image, video, or tabular ML data
- Splitting a dataset into train/val/test, especially with correlated samples
- Investigating validation metrics that look too good to be true
- Choosing between Parquet, LMDB, WebDataset, TFRecord, Arrow, and raw files
- Writing data quality checks or Great Expectations suites
- Processing datasets larger than memory
- Adding or changing fields on an existing dataset

## Key Patterns

### Group-Aware Splitting

Whole groups -- videos, patients, matches, sessions -- go to exactly one split, and the pipeline itself asserts disjointness.

```python
groups = df[key].unique().sample(fraction=1.0, seed=config.seed, shuffle=True)
n_train = int(len(groups) * config.train_ratio)
n_val = int(len(groups) * config.val_ratio)

train_groups = set(groups[:n_train].to_list())
val_groups = set(groups[n_train : n_train + n_val].to_list())
test_groups = set(groups[n_train + n_val :].to_list())

if not train_groups.isdisjoint(test_groups):
    msg = "train/test group overlap — data leakage"
    raise ValueError(msg)
```

### Validated Manifest

```python
class ImageRecord(BaseModel):
    """Validated image metadata record."""

    file_hash: str = Field(min_length=64, max_length=64)  # SHA-256
    relative_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    label: str
    group_id: str | None = None  # patient / video / scene identifier
```

The manifest is written as Parquet with `compression="zstd"`, optionally Hive-partitioned by split so readers can skip whole partitions.

### Storage Format Selection

| Data | Format | Why |
| --- | --- | --- |
| Tabular metadata | Parquet | Columnar, compressed, fast predicate filtering |
| Distributed training | WebDataset | `.tar` shards, sequential throughput, S3-friendly |
| Single-node random access | LMDB | Memory-mapped, zero-copy reads |
| Cross-language interchange | Arrow IPC | Zero-copy, language-agnostic |
| Small or churning datasets | Raw files + Parquet manifest | Simplest and debuggable |

### Schema Evolution

```python
class SchemaV3(SchemaV2):
    """V3 — adds split assignment and quality score."""
    split: str = Field(pattern=r"^(train|val|test)$")
    quality_score: float = Field(ge=0.0, le=1.0, default=1.0)


def migrate_v2_to_v3(record: SchemaV2, split: str, quality_score: float = 1.0) -> SchemaV3:
    """Enrich a V2 record with its split assignment and quality score."""
    return SchemaV3(**record.model_dump(), split=split, quality_score=quality_score)
```

## Anti-Patterns

- Do not random-split a DataFrame when a grouping key exists -- correlated samples leak across train and test
- Do not skip stratification when there is no grouping key -- imbalanced splits make metrics unreliable
- Do not modify raw data in place -- write transformations to a separate directory
- Do not use CSV for large datasets -- Parquet, LMDB, or WebDataset instead
- Do not apply training augmentations to val or test sets
- Do not change a dataset schema silently -- version it and write a migration
- Do not assume a public or "curated" dataset is clean -- run the quality gate anyway
- Do not commit data to Git -- use DVC or object storage

## Combines Well With

- **DVC** -- Version the manifests, splits, and processed outputs this skill produces
- **Pydantic** -- Validated configs and per-record schema enforcement
- **PyTorch Lightning** -- `LightningDataModule` consuming the validated splits
- **Model Evaluation** -- Scoring models on splits built without leakage
- **Testing** -- Unit tests for transforms, integration tests for pipeline stages

## Full Reference

See [`skills/data-pipelines/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/data-pipelines/SKILL.md) for the full decision matrices, Great Expectations suite, augmentation config pattern, chunked processing utilities, and schema registry.

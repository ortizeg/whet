---
name: data-pipelines
description: >
  Use this skill when designing a dataset or ETL/ELT pipeline, choosing a storage format
  (Parquet, LMDB, WebDataset, TFRecord), splitting a dataset into train/val/test, preventing
  data leakage across groups (patient, video, match, player, scene), validating data quality,
  evolving a dataset schema, or processing image/video data at a scale that exceeds memory.
  Reach for it any time you would otherwise glob a directory, random-split a DataFrame, or
  hand-write a CSV manifest, even if the user doesn't say "pipeline" or "ETL" explicitly.
  Not for file-level versioning of datasets and checkpoints (see `dvc`), and not for scoring
  or comparing models (see `model-evaluation`).
---

# Data Pipelines for CV/ML

Reproducible, scalable pipelines that feed ML training and inference.

## Core Principles

1. **Data quality first.** No model compensates for bad data. Validate early and often.
2. **Reproducibility.** Every version, transformation, and split is traceable via content hashes and seeded, deterministic code.
3. **Schema as contract.** Schemas bind pipeline stages together. Changes require explicit migrations, never silent mutations.
4. **Immutable datasets.** Published versions are never modified in place; new versions carry lineage to their source.
5. **Fail fast, fail loud.** Raise on quality violations immediately rather than propagating corrupt data downstream.

```bash
pixi add polars pillow lmdb
pixi add --pypi great-expectations albumentations webdataset
```

## Pipeline Architecture

```
Building a data pipeline?
├── Small (< 10 GB) ............ Polars / Pandas + DVC
├── Medium (10–500 GB) ......... Polars / DuckDB + Parquet + DVC
├── Large (500 GB – 10 TB) ..... Arrow / Spark + cloud object storage
├── Streaming / continuous ..... Kafka / Flink + Delta Lake
└── Annotation pipeline ........ Label Studio / CVAT + validation hooks
```

A pipeline is a fixed sequence of named stages — `extract → validate_raw →
transform → validate_transformed → load` — with a real gate between each.
Configure it with Pydantic so bad parameters fail at load time, not three hours in.

```python
"""Data pipeline configuration."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class PipelineConfig(BaseModel):
    """Top-level data pipeline configuration."""

    name: str = Field(min_length=1)
    source_dir: Path
    output_dir: Path
    storage_format: Literal["parquet", "arrow", "lmdb", "tfrecord", "webdataset"] = "parquet"
    num_workers: int = Field(ge=1, le=64, default=8)
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$", default="1.0.0")

    @field_validator("output_dir")
    @classmethod
    def output_must_differ_from_source(cls, v: Path, info: ValidationInfo) -> Path:
        """Prevent writing transformed data over the raw inputs."""
        if v == info.data.get("source_dir"):
            msg = "output_dir must differ from source_dir"
            raise ValueError(msg)
        return v
```

The anti-pattern this replaces: a bare `process_data(input_path, output_path)` that
globs `*.jpg` and rewrites images with no schema, no logging, and no guard against
clobbering the raw inputs.

## ETL: Building a Dataset Manifest

The manifest is the source of truth: one validated row per sample, carrying a
content hash for deduplication and integrity, written as Parquet (never CSV).

```python
"""ETL for image datasets: hash, validate, write a Parquet manifest."""

import hashlib
from pathlib import Path

import polars as pl
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class ImageRecord(BaseModel):
    """Validated image metadata record."""

    file_hash: str = Field(min_length=64, max_length=64)  # SHA-256
    relative_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    channels: int = Field(ge=1, le=4)
    label: str
    group_id: str | None = None  # patient / video / scene identifier
    file_size_bytes: int = Field(ge=1)


def compute_file_hash(path: Path) -> str:
    """SHA-256 of file contents, for content-addressable storage."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_manifest(paths: list[Path], labels: dict[Path, str], out: Path) -> pl.DataFrame:
    """Read image metadata into validated records and write them as Parquet."""
    records = []
    for path in paths:
        with Image.open(path) as img:
            width, height, channels = *img.size, len(img.getbands())
        records.append(
            ImageRecord(
                file_hash=compute_file_hash(path),
                relative_path=str(path),
                width=width,
                height=height,
                channels=channels,
                label=labels[path],
                file_size_bytes=path.stat().st_size,
            ).model_dump()
        )
    df = pl.DataFrame(records)
    df.write_parquet(out, compression="zstd")
    logger.info("Manifest written: {} records to {}", len(df), out)
    return df
```

Hashes deduplicate byte-identical images that would otherwise straddle splits and
detect corruption between runs. `split` is deliberately *not* a manifest field: it
is assigned downstream, so a dataset can be re-split without re-extracting.

## Dataset Splitting Without Leakage

**This is the highest-risk step in any CV pipeline.** Correlated samples — frames
from the same video, images of the same patient, plays from the same match, crops
of the same player — are near-duplicates. If they straddle train/val/test,
validation measures memorization rather than generalization and the model fails
silently in production. Split on the *group*, never on the row, whenever a grouping
key exists: video/clip ID, patient/subject ID, match/session ID, player identity,
camera or site ID, or capture date for time-correlated data. Stratify only when no
such key exists.

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

Anti-pattern — a plain random split silently leaks every group:

```python
# WRONG: neighbouring frames of the same clip land in both train and test.
train, val, test = np.split(df.sample(frac=1), [int(0.7 * len(df)), int(0.85 * len(df))])
```

Group-aware splits skew ratios when group sizes are uneven. Always log realized
per-split record counts and class distribution and check them before training;
rebalance by greedily assigning the largest groups first if needed.

## Storage Format Selection

```
Choosing a storage format?
├── Tabular metadata (labels, splits, paths, hashes)
│   └── Parquet ....... columnar, compressed, fast predicate filtering (Polars/DuckDB)
├── Streaming large image/video datasets
│   ├── WebDataset .... .tar shards; distributed/multi-node, S3-friendly, no random access
│   └── TFRecord ...... TensorFlow ecosystem, sequential reads
├── Random-access image datasets (single node, reshuffled every epoch)
│   └── LMDB .......... memory-mapped zero-copy reads, single-writer, bad over NFS
├── In-memory analytics / cross-language interchange
│   └── Arrow IPC ..... zero-copy, language-agnostic
└── Small datasets (< 1 GB) or active annotation churn
    └── Raw folders + Parquet manifest ..... simplest, debuggable, diffable
```

Rules of thumb: raw files until random reads become the bottleneck; LMDB for
single-node random access; WebDataset once training spans nodes or lives in object
storage; Parquet for everything tabular, always. Write manifests with Hive-style
partitions (`split=train/data.parquet`, `compression="zstd"`,
`row_group_size=10_000`) so readers can skip whole splits.

```python
"""LMDB packing for fast random-access image reads."""

from pathlib import Path

import lmdb
from loguru import logger


def build_lmdb_dataset(image_paths: list[Path], out_path: Path, map_gb: int = 50) -> None:
    """Pack encoded image bytes into LMDB keyed by zero-padded index."""
    env = lmdb.open(str(out_path), map_size=map_gb * 1024**3)
    with env.begin(write=True) as txn:
        for idx, img_path in enumerate(image_paths):
            txn.put(f"{idx:08d}".encode(), img_path.read_bytes())
        txn.put(b"__len__", str(len(image_paths)).encode())
    env.close()
    logger.info("Built LMDB: {} images at {}", len(image_paths), out_path)
```

## Data Quality Validation

Two complementary layers: Pydantic validates each record at construction time;
Great Expectations validates the dataset in aggregate — distributions, uniqueness,
ranges — and emits a durable report artifact.

```python
"""Great Expectations suite for an image dataset manifest."""

import great_expectations as gx
from loguru import logger

COLUMNS = ["file_hash", "relative_path", "width", "height", "channels", "label", "group_id"]


def create_image_dataset_expectations(context: gx.DataContext) -> None:
    """Define the expectation suite for an image classification manifest."""
    suite = context.add_expectation_suite("image_classification_suite")
    gxe = gx.expectations
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(column_set=COLUMNS),
        *(gxe.ExpectColumnValuesToNotBeNull(column=c) for c in ("file_hash", "label")),
        *(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=32, max_value=8192)
          for c in ("width", "height")),
        # Duplicate hashes mean duplicate images — a cross-split leakage vector.
        gxe.ExpectColumnValuesToBeUnique(column="file_hash"),
    ]
    for expectation in expectations:
        suite.add_expectation(expectation)
    logger.info("Created suite with {} expectations", len(expectations))
```

Mirror the cheapest checks as an in-pipeline gate — null counts per column,
`df["file_hash"].is_duplicated().sum()`, per-class counts from
`df.group_by("label").len()`, and a filter for undersized images — returning a
frozen Pydantic `QualityReport` and raising when it fails. Run the gate after
extraction *and* after splitting: a healthy manifest still yields empty or
single-class splits when ratios or group sizes are wrong.

## Augmentation Pipeline Design

Augmentation is part of the data contract, not a training detail: configure it,
scale it with a single `strength` knob, and keep val/test strictly deterministic.

```python
"""Albumentations pipelines driven by config."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel, Field

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class AugmentationConfig(BaseModel):
    image_size: int = Field(ge=32, default=640)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)


def build_train_transforms(cfg: AugmentationConfig) -> A.Compose:
    """Training augmentations, all probabilities scaled by a single strength knob."""
    s, size = cfg.strength, cfg.image_size
    return A.Compose(
        [
            A.RandomResizedCrop(height=size, width=size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2 * s, contrast=0.2 * s, saturation=0.2 * s, p=0.8),
            A.GaussNoise(p=0.3 * s),
            A.CoarseDropout(max_holes=int(8 * s), p=0.3 * s),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
```

Val/test transforms are the same `Compose` with everything random removed —
`A.Resize`, `A.Normalize`, `ToTensorV2`, nothing else. Log the augmentation config
with the run: an unrecorded `strength` change is an unreproducible run.

## Large-Scale Processing

Chunked parallel processing with per-item error isolation and progress logging;
streaming reads for datasets larger than RAM.

```python
"""Parallel chunked processing and streaming reads."""

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def process_in_chunks(
    paths: list[Path], fn: Callable[[Path], Any], workers: int = 8, chunk: int = 1000
) -> list[Any]:
    """Process files in parallel chunks; one failed item never kills the run."""
    results: list[Any] = []
    total = len(paths)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0, total, chunk):
            futures = {executor.submit(fn, p): p for p in paths[start : start + chunk]}
            for future in as_completed(futures):
                try:
                    results.append(future.result(timeout=300))
                except Exception:
                    logger.exception("Failed to process: {}", futures[future])
            done = min(start + chunk, total)
            logger.info("Progress: {}/{} ({:.1f}%)", done, total, done / total * 100)
    logger.info("Processed {}/{} files successfully", len(results), total)
    return results
```

For datasets larger than RAM, never call `pl.read_parquet`. Scan lazily and pull
fixed-size slices: `reader = pl.scan_parquet(path)`, then
`reader.slice(offset, batch).collect()` per batch, with the row count from
`reader.select(pl.len()).collect().item()`.

## Schema Evolution and Migration

Never mutate a dataset schema in place. Declare each version as its own model,
register it, and provide an explicit migration function per hop.

```python
"""Versioned dataset schemas with an explicit migration chain."""

from pydantic import BaseModel, Field


class SchemaV1(BaseModel):
    """Original — path and label only."""
    image_path: str
    label: str


class SchemaV2(SchemaV1):
    """V2 — adds dimensions and content hash."""
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    file_hash: str


class SchemaV3(SchemaV2):
    """V3 — adds split assignment and quality score."""
    split: str = Field(pattern=r"^(train|val|test)$")
    quality_score: float = Field(ge=0.0, le=1.0, default=1.0)


def migrate_v2_to_v3(record: SchemaV2, split: str, quality_score: float = 1.0) -> SchemaV3:
    """Enrich a V2 record with its split assignment and quality score."""
    return SchemaV3(**record.model_dump(), split=split, quality_score=quality_score)


# Ordered registry: reading version N and targeting M applies hops N..M in order.
SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "1.0.0": SchemaV1,
    "2.0.0": SchemaV2,
    "3.0.0": SchemaV3,
}
MIGRATIONS = {("2.0.0", "3.0.0"): migrate_v2_to_v3}
```

Subclassing keeps additive versions short, but write a standalone model whenever a
field changes meaning or type — inheritance must never hide a breaking change.
Store `schema_version` alongside the data (Parquet key-value metadata or a sidecar
`dataset.json`) so readers dispatch to the right model instead of guessing.

## Versioning

Version datasets, manifests, and split outputs with DVC — see the `dvc` skill.

## Anti-Patterns

- **Never split randomly when a grouping key exists.** Frames from one video, images of one patient, or plays from one match must live in exactly one split.
- **Never split without stratification** when no grouping key exists — class imbalance across splits makes metrics unreliable.
- **Never modify raw data in place.** Write transformations to a separate directory; keep originals intact.
- **Never skip validation between pipeline stages.** Catching corrupt data early saves hours of wasted training, and no public or "curated" dataset is clean by default.
- **Never use CSV for large datasets.** Parquet for tabular, LMDB for random-access images, WebDataset for streaming.
- **Never apply training augmentations to val or test.** Only deterministic resize/normalize outside training.
- **Never silently change a schema.** Version it and write an explicit migration.
- **Never hard-code paths or ratios.** Put them in a validated Pydantic config.
- **Never version data in Git.** Use DVC or object storage.

## Related Skills

- `dvc` — dataset and artifact versioning, pipeline reproducibility, remotes
- `pydantic` — validated configs and schema definitions for every stage
- `pytorch-lightning` — `LightningDataModule` consuming validated splits
- `model-evaluation` — scoring models on the splits this skill produces
- `testing` — unit tests for transforms, integration tests for pipeline stages

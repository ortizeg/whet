# Pipeline Architecture and ETL

Sizing the pipeline, configuring it with Pydantic, and building the Parquet dataset manifest that everything downstream reads.

## Contents

- [Pipeline Architecture](#pipeline-architecture)
- [PipelineConfig](#pipelineconfig)
- [ETL: Building a Dataset Manifest](#etl-building-a-dataset-manifest)
- [Why Hashes, and Why No `split` Column](#why-hashes-and-why-no-split-column)
- [Dependencies](#dependencies)

## Pipeline Architecture

```
Building a data pipeline?
├── Small (< 10 GB) ............ Polars / Pandas
├── Medium (10–500 GB) ......... Polars / DuckDB + Parquet
├── Large (500 GB – 10 TB) ..... Arrow / Spark + cloud object storage
├── Streaming / continuous ..... Kafka / Flink + Delta Lake
└── Annotation pipeline ........ Label Studio / CVAT + validation hooks
```

A pipeline is a fixed sequence of named stages — `extract → validate_raw →
transform → validate_transformed → load` — with a real gate between each.
Configure it with Pydantic so bad parameters fail at load time, not three hours in.

## PipelineConfig

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

## Why Hashes, and Why No `split` Column

Hashes deduplicate byte-identical images that would otherwise straddle splits and
detect corruption between runs. `split` is deliberately *not* a manifest field: it
is assigned downstream, so a dataset can be re-split without re-extracting.

## Dependencies

```bash
pixi add polars pillow lmdb
pixi add --pypi great-expectations albumentations webdataset
```

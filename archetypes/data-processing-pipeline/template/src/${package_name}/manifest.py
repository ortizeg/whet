"""The dataset manifest — one validated, content-hashed row per sample.

The manifest is the source of truth for the dataset: bulk data lives in object
storage, the manifest (and its fingerprint) is what gets versioned. Note that
``split`` is deliberately *not* a manifest column — it is assigned downstream so
a dataset can be re-split without re-extracting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ${package_name}.config import Compression

SCHEMA_VERSION = 1
"""Bump on any breaking manifest schema change and write an explicit migration."""

HASH_CHUNK_BYTES = 1 << 20

MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "record_id": pl.String(),
    "path": pl.String(),
    "label": pl.String(),
    "group_id": pl.String(),
    "content_sha256": pl.String(),
    "size_bytes": pl.Int64(),
    "schema_version": pl.Int32(),
}


class ImageRecord(BaseModel, frozen=True):
    """A single validated sample in the dataset.

    ``group_id`` is the leakage-critical field: samples sharing a ``group_id``
    (frames of one clip, images of one patient) must never straddle splits.
    """

    record_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    label: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    content_sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    schema_version: int = SCHEMA_VERSION


def sha256_file(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    """Content-hash a file in chunks so arbitrarily large files stream."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def make_record(path: Path, label: str, group_id: str, root: Path) -> ImageRecord:
    """Build one manifest record from a file on disk."""
    content_sha256 = sha256_file(path)
    return ImageRecord(
        record_id=content_sha256[:16],
        path=path.relative_to(root).as_posix(),
        label=label,
        group_id=group_id,
        content_sha256=content_sha256,
        size_bytes=path.stat().st_size,
    )


def build_manifest(records: Sequence[ImageRecord]) -> pl.DataFrame:
    """Turn validated records into a manifest DataFrame with an explicit schema."""
    rows = [record.model_dump() for record in records]
    frame = pl.DataFrame(rows, schema=MANIFEST_SCHEMA)
    logger.info("Built manifest: {} records, schema v{}", frame.height, SCHEMA_VERSION)
    return frame


def manifest_fingerprint(manifest: pl.DataFrame) -> str:
    """Deterministic content hash of the whole dataset.

    Order-independent: hashing the sorted per-record content hashes means two
    pipeline runs over the same bytes produce the same fingerprint, whatever
    order the filesystem walked them in.
    """
    hashes = sorted(manifest["content_sha256"].to_list())
    digest = hashlib.sha256(f"v{SCHEMA_VERSION}".encode())
    for value in hashes:
        digest.update(value.encode())
    return digest.hexdigest()


def write_manifest(
    manifest: pl.DataFrame,
    path: Path,
    compression: Compression = "zstd",
) -> Path:
    """Write the manifest as Parquet (never CSV) and log where it went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(path, compression=compression)
    logger.info("Wrote manifest: {} ({} records)", path, manifest.height)
    return path


def read_manifest(path: Path) -> pl.DataFrame:
    """Read a manifest and enforce the schema contract."""
    manifest = pl.read_parquet(path)
    missing = [column for column in MANIFEST_SCHEMA if column not in manifest.columns]
    if missing:
        msg = f"Manifest {path} is missing required columns: {missing}"
        raise ValueError(msg)
    versions = set(manifest["schema_version"].to_list())
    if versions - {SCHEMA_VERSION}:
        msg = f"Manifest {path} has schema versions {sorted(versions)}, expected {SCHEMA_VERSION}"
        raise ValueError(msg)
    logger.info("Read manifest: {} ({} records)", path, manifest.height)
    return manifest


def write_dataset_card(
    path: Path,
    manifest: pl.DataFrame,
    splits: Iterable[tuple[str, int]],
    extra: dict[str, object] | None = None,
) -> Path:
    """Write the small, Git-committable JSON sidecar describing this dataset version."""
    card: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": manifest_fingerprint(manifest),
        "n_records": manifest.height,
        "n_groups": manifest["group_id"].n_unique(),
        "labels": sorted(manifest["label"].unique().to_list()),
        "splits": dict(splits),
    }
    if extra:
        card.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote dataset card: {} (fingerprint {})", path, card["fingerprint"])
    return path

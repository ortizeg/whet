"""Manifest schema, content hashing, and Parquet round-tripping."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from ${package_name}.manifest import (
    MANIFEST_SCHEMA,
    SCHEMA_VERSION,
    ImageRecord,
    build_manifest,
    make_record,
    manifest_fingerprint,
    read_manifest,
    sha256_file,
    write_dataset_card,
    write_manifest,
)


def test_manifest_has_the_declared_schema(manifest: pl.DataFrame) -> None:
    assert manifest.columns == list(MANIFEST_SCHEMA)
    assert manifest["schema_version"].unique().to_list() == [SCHEMA_VERSION]


def test_empty_manifest_still_has_a_schema() -> None:
    empty = build_manifest([])
    assert empty.height == 0
    assert empty.columns == list(MANIFEST_SCHEMA)


def test_record_rejects_a_malformed_hash() -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            record_id="abc",
            path="a/b.png",
            label="goal",
            group_id="clip",
            content_sha256="tooshort",
            size_bytes=1,
        )


def test_sha256_file_matches_make_record(tmp_path: Path) -> None:
    path = tmp_path / "goal" / "clip_000" / "frame_0000.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"pixels")

    record = make_record(path, label="goal", group_id="clip_000", root=tmp_path)
    assert record.content_sha256 == sha256_file(path)
    assert record.path == "goal/clip_000/frame_0000.png"
    assert record.size_bytes == 6


def test_fingerprint_is_order_independent(manifest: pl.DataFrame) -> None:
    shuffled = manifest.sample(fraction=1.0, shuffle=True, seed=7)
    assert manifest_fingerprint(shuffled) == manifest_fingerprint(manifest)


def test_fingerprint_changes_when_content_changes(manifest: pl.DataFrame) -> None:
    mutated = manifest.with_columns(
        pl.when(pl.col("record_id") == manifest["record_id"][0])
        .then(pl.lit("f" * 64))
        .otherwise(pl.col("content_sha256"))
        .alias("content_sha256")
    )
    assert manifest_fingerprint(mutated) != manifest_fingerprint(manifest)


def test_manifest_round_trips_through_parquet(manifest: pl.DataFrame, tmp_path: Path) -> None:
    path = write_manifest(manifest, tmp_path / "manifest.parquet")
    restored = read_manifest(path)
    assert restored.height == manifest.height
    assert manifest_fingerprint(restored) == manifest_fingerprint(manifest)


def test_read_manifest_rejects_a_missing_column(manifest: pl.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "broken.parquet"
    manifest.drop("group_id").write_parquet(path)
    with pytest.raises(ValueError, match="missing required columns"):
        read_manifest(path)


def test_read_manifest_rejects_an_unknown_schema_version(
    manifest: pl.DataFrame, tmp_path: Path
) -> None:
    path = tmp_path / "future.parquet"
    bumped = pl.lit(SCHEMA_VERSION + 1).cast(pl.Int32).alias("schema_version")
    manifest.with_columns(bumped).write_parquet(path)
    with pytest.raises(ValueError, match="schema versions"):
        read_manifest(path)


def test_dataset_card_records_the_fingerprint(manifest: pl.DataFrame, tmp_path: Path) -> None:
    path = write_dataset_card(tmp_path / "dataset.json", manifest, [("train", 10), ("test", 4)])
    card = json.loads(path.read_text())
    assert card["fingerprint"] == manifest_fingerprint(manifest)
    assert card["n_records"] == manifest.height
    assert card["splits"] == {"train": 10, "test": 4}

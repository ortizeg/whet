"""End-to-end pipeline and CLI tests over a synthetic raw tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ${package_name}.cli import main
from ${package_name}.config import IngestConfig, PipelineConfig, SplitConfig
from ${package_name}.manifest import read_manifest
from ${package_name}.pipeline import run_pipeline
from ${package_name}.sample_data import encode_png
from ${package_name}.stages import stage_ingest


def test_sample_data_is_a_valid_png() -> None:
    data = encode_png(4, 4, (1, 2, 3))
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert data.endswith(b"IEND\xae\x42\x60\x82")


def test_ingest_derives_label_and_group_from_the_layout(raw_root: Path) -> None:
    manifest = stage_ingest(IngestConfig(root=raw_root))
    assert manifest.height == 2 * 4 * 5
    assert sorted(manifest["label"].unique().to_list()) == ["goal", "no_goal"]
    assert manifest["group_id"].n_unique() == 8
    assert manifest["content_sha256"].n_unique() == manifest.height


def test_ingest_skips_unknown_extensions(raw_root: Path) -> None:
    (raw_root / "goal" / "goal_clip_000" / "notes.txt").write_text("ignore me")
    manifest = stage_ingest(IngestConfig(root=raw_root))
    assert manifest.height == 2 * 4 * 5


def test_ingest_requires_an_existing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        stage_ingest(IngestConfig(root=tmp_path / "nope"))


def test_run_pipeline_writes_disjoint_splits(pipeline_config: PipelineConfig) -> None:
    result = run_pipeline(pipeline_config)

    assert result.n_records == 40
    assert result.n_groups == 8
    assert sum(result.split_counts.values()) == result.n_records

    out = pipeline_config.output.processed_dir
    assert (out / "manifest.parquet").exists()
    assert set(result.artifacts) == {"manifest", "train", "val", "test", "card"}

    groups = {
        name: set(read_manifest(out / "splits" / f"{name}.parquet")["group_id"].to_list())
        for name in ("train", "val", "test")
    }
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])

    card = json.loads((out / "dataset.json").read_text())
    assert card["fingerprint"] == result.fingerprint
    assert card["n_groups"] == 8


def test_pipeline_is_reproducible(pipeline_config: PipelineConfig) -> None:
    first = run_pipeline(pipeline_config)
    second = run_pipeline(pipeline_config)
    assert first.fingerprint == second.fingerprint
    assert first.split_counts == second.split_counts


def test_pipeline_aborts_on_duplicate_content(pipeline_config: PipelineConfig) -> None:
    source = pipeline_config.ingest.root / "goal" / "goal_clip_000" / "frame_0000.png"
    clone = pipeline_config.ingest.root / "goal" / "goal_clip_001" / "cloned.png"
    clone.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="Data quality gate failed"):
        run_pipeline(pipeline_config)


def test_pipeline_rejects_a_dataset_with_too_few_groups(
    pipeline_config: PipelineConfig, tmp_path: Path
) -> None:
    thin = PipelineConfig(
        ingest=IngestConfig(root=pipeline_config.ingest.root / "goal" / "goal_clip_000"),
        split=SplitConfig(),
        output=pipeline_config.output,
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        run_pipeline(thin)


def test_cli_sample_then_run(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    config = tmp_path / "pipeline.toml"
    config.write_text(
        "\n".join(
            [
                'name = "cli-test"',
                "[ingest]",
                f'root = "{raw.as_posix()}"',
                "[quality]",
                "min_groups_per_label = 2",
                "[output]",
                f'processed_dir = "{(tmp_path / "processed").as_posix()}"',
            ]
        )
    )

    assert main(["sample", "--root", str(raw), "--groups", "5", "--frames", "3"]) == 0
    assert main(["scan", "--config", str(config)]) == 0
    assert main(["validate", "--config", str(config)]) == 0
    assert main(["run", "--config", str(config)]) == 0
    assert (tmp_path / "processed" / "splits" / "test.parquet").exists()


def test_cli_returns_nonzero_when_the_root_is_missing(tmp_path: Path) -> None:
    config = tmp_path / "pipeline.toml"
    config.write_text(f'[ingest]\nroot = "{(tmp_path / "missing").as_posix()}"\n')
    assert main(["run", "--config", str(config)]) == 1

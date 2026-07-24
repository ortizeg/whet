"""Tests for the install manifest and prune safety."""

from __future__ import annotations

import json
from pathlib import Path

from whet.core.manifest import (
    MANIFEST_NAME,
    find_orphans,
    manifest_path,
    read_manifest,
    write_manifest,
)


def test_read_manifest_missing_returns_empty(tmp_path: Path) -> None:
    """No manifest means whet owns nothing here, so prune may remove nothing."""
    assert read_manifest(tmp_path) == []


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    write_manifest(tmp_path, ["beta", "alpha", "alpha"])
    assert read_manifest(tmp_path) == ["alpha", "beta"]


def test_manifest_is_json_with_version(tmp_path: Path) -> None:
    write_manifest(tmp_path, ["onnx"])
    raw = json.loads(manifest_path(tmp_path).read_text())
    assert raw["version"] == 1
    assert raw["skills"] == ["onnx"]


def test_manifest_file_name(tmp_path: Path) -> None:
    assert manifest_path(tmp_path).name == MANIFEST_NAME


def test_read_manifest_tolerates_corrupt_file(tmp_path: Path) -> None:
    """A damaged manifest must not crash the install, and must not authorize deletion."""
    manifest_path(tmp_path).write_text("{not json")
    assert read_manifest(tmp_path) == []


def test_read_manifest_tolerates_wrong_shape(tmp_path: Path) -> None:
    manifest_path(tmp_path).write_text(json.dumps({"skills": "onnx"}))
    assert read_manifest(tmp_path) == []


def test_find_orphans_detects_deleted_skill() -> None:
    """A skill whet installed that no longer exists upstream is an orphan."""
    assert find_orphans(["onnx", "dvc"], {"onnx"}) == ["dvc"]


def test_find_orphans_ignores_skills_whet_did_not_install() -> None:
    """Skills owned by other tools (GSD, interface-design) must never be pruned."""
    previously_installed = ["onnx"]
    available = {"onnx"}
    # gsd-planner and interface-design live in the same directory but are not in the
    # manifest, so they can never be returned as orphans.
    assert find_orphans(previously_installed, available) == []


def test_find_orphans_ignores_filtered_but_existing_skills() -> None:
    """An 'extra' tier skill skipped this run still exists upstream — not an orphan."""
    assert find_orphans(["onnx", "kubernetes"], {"onnx", "kubernetes"}) == []


def test_find_orphans_is_sorted_and_deduped() -> None:
    assert find_orphans(["b", "a", "a"], set()) == ["a", "b"]

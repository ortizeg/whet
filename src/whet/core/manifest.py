"""Install manifest — records which skills whet installed into a target directory.

Skill directories are shared: a Claude Code skills directory typically also holds
skills installed by other tools (GSD, interface-design, hand-written ones). whet must
therefore never delete a directory it did not create. The manifest is the record of
ownership that makes `whet install --prune` safe.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = ".whet-manifest.json"
MANIFEST_VERSION = 1


def manifest_path(target_dir: Path) -> Path:
    """Path to the manifest for a given install directory."""
    return target_dir / MANIFEST_NAME


def read_manifest(target_dir: Path) -> list[str]:
    """Skill names whet has previously installed here.

    Returns an empty list when no manifest exists (a pre-manifest or first-time
    install), which means prune has nothing it is allowed to remove.
    """
    path = manifest_path(target_dir)
    if not path.is_file():
        return []

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(raw, dict):
        return []

    skills = raw.get("skills")
    if not isinstance(skills, list):
        return []

    return [name for name in skills if isinstance(name, str)]


def write_manifest(target_dir: Path, skill_names: list[str]) -> None:
    """Record the skills whet owns in this directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {"version": MANIFEST_VERSION, "skills": sorted(set(skill_names))}
    manifest_path(target_dir).write_text(json.dumps(payload, indent=2) + "\n")


def find_orphans(previously_installed: list[str], available: set[str]) -> list[str]:
    """Skills whet installed that no longer exist in the source library.

    Only names whet recorded installing are ever considered, so skills owned by other
    tools are invisible to prune. Skills that still exist upstream but were filtered
    out of this run (for example `tier = "extra"` without --include-extras) are NOT
    orphans — they were deliberately skipped, not deleted.
    """
    return sorted(name for name in set(previously_installed) if name not in available)

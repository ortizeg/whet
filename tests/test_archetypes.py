"""Test that all archetypes are complete."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib

ARCHETYPES_DIR = Path("archetypes")
SKILLS_DIR = Path("skills")


def get_all_archetypes() -> list[Path]:
    """Get all archetype directories."""
    return sorted(
        [d for d in ARCHETYPES_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_readme(archetype_dir: Path) -> None:
    """Test archetype has README."""
    readme = archetype_dir / "README.md"
    assert readme.exists(), f"Missing README in {archetype_dir.name}"

    content = readme.read_text()
    assert len(content) > 500, f"README too short in {archetype_dir.name}"


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_toml(archetype_dir: Path) -> None:
    """Test archetype has archetype.toml metadata file."""
    toml_path = archetype_dir / "archetype.toml"
    assert toml_path.exists(), f"Missing archetype.toml in {archetype_dir.name}"


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_structure(archetype_dir: Path) -> None:
    """Test archetype has expected structure."""
    has_template = (archetype_dir / "template").exists()
    readme = archetype_dir / "README.md"
    has_structure_doc = "```" in readme.read_text() if readme.exists() else False

    assert has_template or has_structure_doc, f"Archetype {archetype_dir.name} missing structure"


def test_minimum_archetype_count() -> None:
    """Test that we have at least the expected number of archetypes."""
    archetypes = get_all_archetypes()
    assert len(archetypes) >= 6, f"Expected at least 6 archetypes, found {len(archetypes)}"


def _archetype_skills(archetype_dir: Path) -> dict[str, list[str]]:
    """Parse the [skills] table from an archetype.toml."""
    with open(archetype_dir / "archetype.toml", "rb") as f:
        raw = tomllib.load(f)
    skills = raw.get("skills", {})
    return {
        key: [s for s in skills.get(key, []) if isinstance(s, str)]
        for key in ("required", "recommended")
    }


def _available_skills() -> set[str]:
    return {d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()}


def _extra_tier_skills() -> set[str]:
    extras: set[str] = set()
    for toml_path in SKILLS_DIR.glob("*/skill.toml"):
        with open(toml_path, "rb") as f:
            raw = tomllib.load(f)
        if raw.get("skill", {}).get("tier") == "extra":
            extras.add(toml_path.parent.name)
    return extras


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_skills_resolve(archetype_dir: Path) -> None:
    """Every skill an archetype lists must actually exist.

    Without this, deleting a skill silently leaves archetypes pointing at nothing —
    and adding one is never noticed. This is the check that was missing while
    cv-inference-service shipped without the fastapi skill.
    """
    available = _available_skills()
    listed = _archetype_skills(archetype_dir)

    for key, names in listed.items():
        dangling = sorted(n for n in names if n not in available)
        assert not dangling, f"{archetype_dir.name} [skills].{key} references missing: {dangling}"


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_does_not_require_extra_tier_skill(archetype_dir: Path) -> None:
    """An archetype must not force an opt-in skill on every generated project."""
    extras = _extra_tier_skills()
    required = _archetype_skills(archetype_dir)["required"]

    forced = sorted(n for n in required if n in extras)
    assert not forced, (
        f"{archetype_dir.name} requires extra-tier skills {forced}; "
        f"list them under [skills].recommended instead"
    )


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_declares_required_skills(archetype_dir: Path) -> None:
    """An archetype whose whole purpose is composition must compose something."""
    required = _archetype_skills(archetype_dir)["required"]
    assert required, f"{archetype_dir.name} declares no required skills"

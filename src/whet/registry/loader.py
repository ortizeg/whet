"""Scan and parse skills from disk."""

from __future__ import annotations

from pathlib import Path

from whet.core.skill import Skill


def discover_skills(skills_dir: Path) -> list[Skill]:
    """Discover all skills in a directory.

    Scans for subdirectories containing SKILL.md and loads each as a Skill.
    """
    return _discover(skills_dir)


def _discover(base_dir: Path) -> list[Skill]:
    """Discover all SKILL.md-based items in a directory."""
    if not base_dir.is_dir():
        return []

    items: list[Skill] = []
    for child in sorted(base_dir.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            skill_md = child / "SKILL.md"
            if skill_md.exists():
                items.append(Skill.from_directory(child))

    return items


def load_skill(skills_dir: Path, name: str) -> Skill | None:
    """Load a single skill by name."""
    skill_path = skills_dir / name
    if skill_path.is_dir() and (skill_path / "SKILL.md").exists():
        return Skill.from_directory(skill_path)
    return None

"""Skill metadata model."""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel, Field

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib


class SkillDependencies(BaseModel):
    """Skill dependency specification."""

    requires: list[str] = Field(default_factory=list)
    recommends: list[str] = Field(default_factory=list)


class SkillCompatibility(BaseModel):
    """Skill compatibility specification."""

    python: str = ">=3.11"
    libraries: dict[str, str] = Field(default_factory=dict)


class SkillMetadata(BaseModel):
    """Machine-readable skill metadata from skill.toml."""

    name: str
    version: str = "1.0.0"
    category: str = "core"
    tier: str = "core"
    """Install tier: "core" installs by default, "extra" is opt-in via --include-extras."""
    tags: list[str] = Field(default_factory=list)

    dependencies: SkillDependencies = Field(default_factory=SkillDependencies)
    compatibility: SkillCompatibility = Field(default_factory=SkillCompatibility)


class SkillFrontmatter(BaseModel):
    """YAML frontmatter parsed from SKILL.md."""

    name: str
    description: str = ""


class Skill(BaseModel):
    """Complete skill representation combining SKILL.md and skill.toml data."""

    name: str
    description: str = ""
    category: str = "core"
    tags: list[str] = Field(default_factory=list)
    path: Path
    metadata: SkillMetadata | None = None

    @property
    def skill_md_path(self) -> Path:
        """Path to the SKILL.md file."""
        return self.path / "SKILL.md"

    @property
    def skill_toml_path(self) -> Path:
        """Path to the skill.toml file."""
        return self.path / "skill.toml"

    @property
    def has_toml(self) -> bool:
        """Whether this skill has a skill.toml metadata file."""
        return self.skill_toml_path.exists()

    @property
    def tier(self) -> str:
        """Install tier from metadata ("core" or "extra"); defaults to "core"."""
        return self.metadata.tier if self.metadata else "core"

    @property
    def references_dir(self) -> Path:
        """Path to the optional references/ directory (progressive disclosure)."""
        return self.path / "references"

    def reference_files(self) -> list[Path]:
        """Sorted reference files, if this skill uses progressive disclosure."""
        if not self.references_dir.is_dir():
            return []
        return sorted(self.references_dir.glob("*.md"))

    def read_skill_md(self) -> str:
        """Read the full SKILL.md content."""
        return self.skill_md_path.read_text()

    def read_flattened(self) -> str:
        """SKILL.md with every reference file appended.

        Used by platforms that store a skill as a single flat file and therefore
        cannot follow `references/` links.
        """
        parts = [self.read_skill_md()]
        for ref in self.reference_files():
            parts.append(f"\n\n---\n\n# Reference: {ref.stem}\n\n{ref.read_text()}")
        return "".join(parts)

    @classmethod
    def from_directory(cls, path: Path) -> Skill:
        """Load a skill from its directory."""
        name = path.name
        description = ""
        category = "core"
        tags: list[str] = []
        metadata: SkillMetadata | None = None

        # Parse YAML frontmatter from SKILL.md
        skill_md = path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text()
            frontmatter = _parse_frontmatter(content)
            if frontmatter:
                name = frontmatter.get("name", name)
                description = frontmatter.get("description", "")

        # Parse skill.toml if present
        skill_toml = path / "skill.toml"
        if skill_toml.exists():
            with open(skill_toml, "rb") as f:
                raw = tomllib.load(f)
            skill_data = raw.get("skill", {})
            deps_data = raw.get("dependencies", {})
            compat_data = raw.get("compatibility", {})

            metadata = SkillMetadata(
                name=skill_data.get("name", name),
                version=skill_data.get("version", "1.0.0"),
                category=skill_data.get("category", "core"),
                tier=skill_data.get("tier", "core"),
                tags=skill_data.get("tags", []),
                dependencies=SkillDependencies(**deps_data),
                compatibility=SkillCompatibility(**compat_data),
            )
            category = metadata.category
            tags = metadata.tags

        return cls(
            name=name,
            description=description.strip() if isinstance(description, str) else "",
            category=category,
            tags=tags,
            path=path,
            metadata=metadata,
        )


def _parse_frontmatter(content: str) -> dict[str, str] | None:
    """Parse YAML frontmatter from a markdown file.

    Returns None if no frontmatter is found.
    """
    if not content.startswith("---"):
        return None

    lines = content.split("\n")
    end_idx = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return None

    # Simple YAML parsing for flat key-value pairs
    result: dict[str, str] = {}
    current_key = ""
    current_value = ""

    for line in lines[1:end_idx]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if ":" in stripped and not stripped.startswith(" ") and not stripped.startswith("-"):
            # Save previous key-value
            if current_key:
                result[current_key] = current_value.strip()

            key, _, value = stripped.partition(":")
            current_key = key.strip()
            value = value.strip()
            # Handle YAML block scalar indicator
            current_value = "" if value == ">" else value
        elif current_key and stripped:
            # Continuation line for block scalar
            if current_value:
                current_value += " " + stripped
            else:
                current_value = stripped

    # Save last key-value
    if current_key:
        result[current_key] = current_value.strip()

    return result

"""Template rendering engine for project scaffolding."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib


class ArchetypeMetadata(BaseModel, frozen=True):
    """Archetype metadata from archetype.toml."""

    name: str
    version: str = "1.0.0"
    category: str = "core"
    tags: list[str] = Field(default_factory=list)
    description: str = ""


class ArchetypeSkills(BaseModel, frozen=True):
    """Skills associated with an archetype."""

    required: list[str] = Field(default_factory=list)
    recommended: list[str] = Field(default_factory=list)


class Archetype(BaseModel, frozen=True):
    """Complete archetype representation."""

    metadata: ArchetypeMetadata
    skills: ArchetypeSkills = Field(default_factory=ArchetypeSkills)
    path: Path
    template_dir: Path

    @property
    def has_template(self) -> bool:
        """Check if the archetype has template files."""
        if not self.template_dir.is_dir():
            return False
        return any(self.template_dir.iterdir())


def load_archetype(archetype_dir: Path) -> Archetype | None:
    """Load an archetype from its directory."""
    toml_path = archetype_dir / "archetype.toml"
    if not toml_path.exists():
        return None

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    arch_data = raw.get("archetype", {})
    skills_data = raw.get("skills", {})

    metadata = ArchetypeMetadata(
        name=arch_data.get("name", archetype_dir.name),
        version=arch_data.get("version", "1.0.0"),
        category=arch_data.get("category", "core"),
        tags=arch_data.get("tags", []),
        description=arch_data.get("description", ""),
    )

    skills = ArchetypeSkills(
        required=skills_data.get("required", []),
        recommended=skills_data.get("recommended", []),
    )

    return Archetype(
        metadata=metadata,
        skills=skills,
        path=archetype_dir,
        template_dir=archetype_dir / "template",
    )


def discover_archetypes(archetypes_dir: Path) -> list[Archetype]:
    """Discover all archetypes in the archetypes directory."""
    if not archetypes_dir.is_dir():
        return []

    archetypes = []
    for d in sorted(archetypes_dir.iterdir()):
        if d.is_dir():
            arch = load_archetype(d)
            if arch:
                archetypes.append(arch)

    return archetypes


class ScaffoldContext(BaseModel):
    """Variables available during template rendering."""

    project_name: str
    project_slug: str = ""
    package_name: str = ""
    description: str = ""
    author: str = ""
    python_version: str = "3.11"

    def model_post_init(self, __context: object) -> None:
        """Derive slug and package name from project name."""
        if not self.project_slug:
            self.project_slug = self.project_name.lower().replace(" ", "-").replace("_", "-")
        if not self.package_name:
            self.package_name = self.project_slug.replace("-", "_")


def render_template(
    archetype: Archetype,
    output_dir: Path,
    context: ScaffoldContext,
) -> Path:
    """Render an archetype template to the output directory.

    Copies files from the archetype's template directory, performing
    variable substitution on file contents and directory/file names.
    """
    if not archetype.has_template:
        # No template files — create a minimal project structure
        return _create_minimal_project(output_dir, context)

    output_dir.mkdir(parents=True, exist_ok=True)
    substitutions = context.model_dump()

    for src_path in archetype.template_dir.rglob("*"):
        if src_path.is_dir():
            continue

        # Compute relative path with variable substitution in names
        rel = src_path.relative_to(archetype.template_dir)
        dest_rel = Path(Template(str(rel)).safe_substitute(substitutions))
        dest_path = output_dir / dest_rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Substitute variables in text files
        if _is_text_file(src_path):
            content = src_path.read_text()
            rendered = Template(content).safe_substitute(substitutions)
            dest_path.write_text(rendered)
        else:
            shutil.copy2(src_path, dest_path)

    return output_dir


def _create_minimal_project(output_dir: Path, context: ScaffoldContext) -> Path:
    """Create a minimal project when no template exists."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create src layout
    pkg_dir = output_dir / "src" / context.package_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text(f'"""{context.project_name}."""\n')

    # Create tests
    tests_dir = output_dir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")

    # Create pyproject.toml
    pyproject = f"""[project]
name = "{context.project_slug}"
version = "0.1.0"
description = "{context.description}"
authors = [{{name = "{context.author}"}}]
requires-python = ">={context.python_version}"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{context.package_name}"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "S", "B", "A", "C4", "T20", "SIM"]
ignore = ["S101"]

[tool.mypy]
python_version = "3.11"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
"""
    (output_dir / "pyproject.toml").write_text(pyproject)

    # Create README
    readme = f"""# {context.project_name}

{context.description}

## Setup

```bash
uv sync
```

## Development

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy src/ --strict
```
"""
    (output_dir / "README.md").write_text(readme)

    return output_dir


def _is_text_file(path: Path) -> bool:
    """Check if a file is likely text (for template substitution)."""
    text_extensions = {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        # Notebooks are JSON, so substitution is safe and lets a template notebook
        # import from `${package_name}` directly instead of resolving it at runtime.
        ".ipynb",
        ".cfg",
        ".ini",
        ".sh",
        ".bash",
        ".dockerfile",
        ".gitignore",
        ".env",
    }
    if path.suffix.lower() in text_extensions:
        return True
    return path.name.lower() in {"dockerfile", "makefile", "justfile", ".gitignore", ".env"}

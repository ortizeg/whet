"""Integration tests for project generation concepts."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_skills_reference_valid_skills() -> None:
    """Test that skill cross-references point to existing skills."""
    skills_dir = Path("skills")
    all_skills = {d.name for d in skills_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}

    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir() or skill_dir.name.startswith("."):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        content = skill_md.read_text()
        # Check for backtick-quoted skill references
        for skill_name in all_skills:
            if f"`{skill_name}`" in content:
                assert (skills_dir / skill_name).exists(), (
                    f"Skill {skill_dir.name} references non-existent skill: {skill_name}"
                )


def test_archetypes_have_consistent_docs() -> None:
    """Test all archetypes document their structure."""
    archetypes_dir = Path("archetypes")

    for arch_dir in archetypes_dir.iterdir():
        if not arch_dir.is_dir() or arch_dir.name.startswith("."):
            continue

        readme = arch_dir / "README.md"
        assert readme.exists(), f"{arch_dir.name} missing README"

        content = readme.read_text()
        # All archetypes should document their directory structure
        assert "```" in content, f"{arch_dir.name} README missing directory structure"
        # All should mention configuration variables
        assert any(
            word in content.lower() for word in ["variable", "config", "template", "customize"]
        ), f"{arch_dir.name} README missing configuration docs"


def test_docs_navigation_references_exist() -> None:
    """Test that docs pages referenced in mkdocs.yml exist."""
    mkdocs_path = Path("mkdocs.yml")
    if not mkdocs_path.exists():
        pytest.skip("mkdocs.yml not found")

    try:
        import yaml
    except ImportError:
        pytest.skip("PyYAML not installed")

    config = yaml.safe_load(mkdocs_path.read_text())
    nav = config.get("nav", [])

    def extract_paths(nav_items: list[object]) -> list[str]:
        """Extract all file paths from nav structure."""
        paths = []
        for item in nav_items:
            if isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str):
                        paths.append(value)
                    elif isinstance(value, list):
                        paths.extend(extract_paths(value))
        return paths

    doc_paths = extract_paths(nav)
    docs_dir = Path("docs")

    missing = []
    for doc_path in doc_paths:
        full_path = docs_dir / doc_path
        if not full_path.exists():
            missing.append(doc_path)

    assert not missing, f"Missing doc pages referenced in mkdocs.yml: {missing}"

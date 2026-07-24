"""Enforce Anthropic's skill-authoring rules so the library cannot rot back.

Every check here corresponds to a rule the library once broke. The completeness
tests assert a skill *exists and is non-trivial*; these assert it is still a
well-formed skill — short enough to load cheaply, discoverable, and with its
progressive-disclosure links intact.

Sources for the rules:
  https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices
  https://code.claude.com/docs/en/skills
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SKILLS_DIR = Path("skills")

# Anthropic's documented ceiling for a SKILL.md body.
MAX_SKILL_LINES = 500
# A reference long enough that a partial read would miss content needs a TOC.
TOC_REQUIRED_OVER_LINES = 100
# Hard cap on the description field; it is the entire trigger surface.
MAX_DESCRIPTION_CHARS = 1024
# Phrases that make a description trigger-first rather than a topic summary.
TRIGGER_PHRASES = ("use this skill when", "use when", "use this when", "reach for it")


def all_skills() -> list[Path]:
    return sorted(d for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists())


def _frontmatter(skill_md: Path) -> str:
    text = skill_md.read_text()
    parts = text.split("---")
    return parts[1] if len(parts) > 2 else ""


def _description(skill_md: Path) -> str:
    match = re.search(r"description:\s*>?\s*(.*?)(?:\n[a-z_]+:|\Z)", _frontmatter(skill_md), re.S)
    return " ".join(match.group(1).split()) if match else ""


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_skill_body_under_line_ceiling(skill_dir: Path) -> None:
    """A SKILL.md loads whole and stays resident, so every line is a recurring cost.

    Nineteen of thirty skills once exceeded this, averaging 551 lines. Nothing
    caught it because the only length assertion was a >500 *character* floor.
    """
    lines = len((skill_dir / "SKILL.md").read_text().splitlines())
    assert lines <= MAX_SKILL_LINES, (
        f"{skill_dir.name}/SKILL.md is {lines} lines (limit {MAX_SKILL_LINES}). "
        f"Move detail into references/ and leave an index — see CLAUDE.md."
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_reference_links_resolve(skill_dir: Path) -> None:
    """Every `references/...` a SKILL.md cites must exist.

    This is the failure mode that made an external UI skill's deep dives 404:
    the body advertised reference files that were never shipped.
    """
    body = (skill_dir / "SKILL.md").read_text()
    cited = set(re.findall(r"`(references/[\w./-]+\.md)`", body))
    missing = sorted(c for c in cited if not (skill_dir / c).is_file())
    assert not missing, (
        f"{skill_dir.name}/SKILL.md cites reference files that do not exist: {missing}"
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_no_orphaned_reference_files(skill_dir: Path) -> None:
    """A reference the index never mentions is unreachable — it can only cost tokens."""
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return

    body = (skill_dir / "SKILL.md").read_text()
    orphans = sorted(r.name for r in refs_dir.glob("*.md") if r.name not in body)
    assert not orphans, (
        f"{skill_dir.name} has reference files nothing links to: {orphans}. "
        f"Add them to the '## Deep dives' list or delete them."
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_references_are_one_level_deep(skill_dir: Path) -> None:
    """SKILL.md -> reference is fine; reference -> reference is not.

    Nested references make the model partial-read and miss content, so Anthropic
    requires a flat single hop.
    """
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return

    nested = sorted(r.name for r in refs_dir.glob("*.md") if "references/" in r.read_text())
    assert not nested, (
        f"{skill_dir.name} reference files link to other references: {nested}; "
        f"keep them one level deep"
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_long_references_have_table_of_contents(skill_dir: Path) -> None:
    """A long reference needs a TOC so a partial read still reveals its scope."""
    refs_dir = skill_dir / "references"
    if not refs_dir.is_dir():
        return

    missing: list[str] = []
    for ref in sorted(refs_dir.glob("*.md")):
        lines = ref.read_text().splitlines()
        if len(lines) <= TOC_REQUIRED_OVER_LINES:
            continue
        head = "\n".join(lines[:25]).lower()
        if "## contents" not in head and not re.search(r"^- \[", head, re.M):
            missing.append(f"{ref.name} ({len(lines)} lines)")

    assert not missing, (
        f"{skill_dir.name} references over {TOC_REQUIRED_OVER_LINES} lines need a "
        f"'## Contents' table of contents: {missing}"
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_description_is_trigger_first(skill_dir: Path) -> None:
    """The description is the entire discovery mechanism, so it must say WHEN.

    Descriptions were once topic summaries ("Covers X, Y, Z"), which under-trigger:
    they describe the subject without ever stating the situation that should fire.
    """
    description = _description(skill_dir / "SKILL.md").lower()
    assert description, f"{skill_dir.name} has an empty description"
    assert any(p in description for p in TRIGGER_PHRASES), (
        f"{skill_dir.name} description is not trigger-first. Lead with when to use it, "
        f'e.g. "Use this skill when ...". Got: {description[:90]}...'
    )


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_description_is_third_person(skill_dir: Path) -> None:
    """First/second-person descriptions hurt discovery; they go into the system prompt.

    Quoted user utterances are stripped first — a description that lists what a user
    might *say* ("set up my editor") is doing keyword coverage, not speaking in first
    person, and that phrasing is encouraged.
    """
    description = _description(skill_dir / "SKILL.md")
    unquoted = re.sub(r'"[^"]*"', "", description)
    bad = re.findall(r"\b(I can|I will|I help|we help|I'll)\b", unquoted, re.I)
    assert not bad, f"{skill_dir.name} description is not third person: {sorted(set(bad))}"


@pytest.mark.parametrize("skill_dir", all_skills(), ids=lambda d: d.name)
def test_description_within_listing_budget(skill_dir: Path) -> None:
    """Descriptions share a capped listing budget; oversized ones get truncated.

    Claude Code truncates the skill listing when it overflows, silently dropping
    the keywords a skill needs to match on.
    """
    length = len(_description(skill_dir / "SKILL.md"))
    assert length <= MAX_DESCRIPTION_CHARS, (
        f"{skill_dir.name} description is {length} chars (cap {MAX_DESCRIPTION_CHARS})"
    )


def test_split_skills_advertise_their_deep_dives() -> None:
    """A skill with references/ should route to them from the index."""
    missing = [
        d.name
        for d in all_skills()
        if (d / "references").is_dir() and "references/" not in (d / "SKILL.md").read_text()
    ]
    assert not missing, f"skills with references/ but no links to them: {missing}"

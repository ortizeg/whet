"""GitHub Copilot adapter — aggregates skills into a single instructions file."""

from __future__ import annotations

from pathlib import Path

from whet.core.skill import Skill


class CopilotAdapter:
    """Adapter for GitHub Copilot.

    Copilot uses a single .github/copilot-instructions.md file.
    Skills are aggregated into sections within this file.
    """

    INSTRUCTIONS_FILE = "copilot-instructions.md"
    MARKER_START = "<!-- whet:start -->"
    MARKER_END = "<!-- whet:end -->"

    @property
    def name(self) -> str:
        return "copilot"

    def install_skill(self, skill: Skill, target_dir: Path) -> Path:
        """Append skill content to the Copilot instructions file.

        Note: For Copilot, use install_skills() to batch-install,
        as each skill is a section in a single file.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / self.INSTRUCTIONS_FILE
        # Copilot stores one flat file, so inline references rather than lose them
        content = skill.read_flattened()
        content = _strip_frontmatter(content)

        start = f"<!-- whet:{skill.name}:start -->"
        end = f"<!-- whet:{skill.name}:end -->"
        section = f"\n\n{start}\n{content}\n{end}\n"

        if dest.exists():
            existing = dest.read_text()
            # Replace existing section or append
            start_marker = f"<!-- whet:{skill.name}:start -->"
            end_marker = f"<!-- whet:{skill.name}:end -->"
            if start_marker in existing:
                before = existing[: existing.index(start_marker)]
                after = existing[existing.index(end_marker) + len(end_marker) :]
                dest.write_text(before + section.strip() + after)
            else:
                dest.write_text(existing + section)
        else:
            header = f"{self.MARKER_START}\n# AI Coding Skills (managed by whet)\n"
            dest.write_text(header + section + f"\n{self.MARKER_END}\n")

        return dest

    def remove_skill(self, skill_name: str, target_dir: Path) -> bool:
        """Remove a skill section from the Copilot instructions file."""
        dest = target_dir / self.INSTRUCTIONS_FILE
        if not dest.exists():
            return False

        content = dest.read_text()
        start_marker = f"<!-- whet:{skill_name}:start -->"
        end_marker = f"<!-- whet:{skill_name}:end -->"

        if start_marker not in content:
            return False

        before = content[: content.index(start_marker)]
        after = content[content.index(end_marker) + len(end_marker) :]
        dest.write_text(before.rstrip() + after.lstrip("\n"))
        return True

    def list_installed(self, target_dir: Path) -> list[str]:
        """List installed skills by scanning for whet markers."""
        dest = target_dir / self.INSTRUCTIONS_FILE
        if not dest.exists():
            return []

        content = dest.read_text()
        names: list[str] = []
        for line in content.split("\n"):
            if line.strip().startswith("<!-- whet:") and line.strip().endswith(":start -->"):
                name = line.strip().removeprefix("<!-- whet:").removesuffix(":start -->")
                names.append(name)
        return sorted(names)


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return content
    lines = content.split("\n")
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return content

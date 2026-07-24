"""Claude Code adapter — native SKILL.md support."""

from __future__ import annotations

import shutil
from pathlib import Path

from whet.core.skill import Skill


class ClaudeAdapter:
    """Adapter for Claude Code.

    Claude Code natively uses SKILL.md files, so the adapter simply copies
    the skill directory to the target location.
    """

    @property
    def name(self) -> str:
        return "claude"

    def install_skill(self, skill: Skill, target_dir: Path) -> Path:
        """Copy the skill directory to Claude's skills location."""
        dest = target_dir / skill.name
        dest.mkdir(parents=True, exist_ok=True)

        # Copy SKILL.md (the primary file the LLM reads)
        src_skill_md = skill.skill_md_path
        if src_skill_md.exists():
            shutil.copy2(src_skill_md, dest / "SKILL.md")

        # Copy README.md if present
        src_readme = skill.path / "README.md"
        if src_readme.exists():
            shutil.copy2(src_readme, dest / "README.md")

        # Copy references/ so the SKILL.md deep-dive links resolve after install
        if skill.references_dir.is_dir():
            shutil.copytree(skill.references_dir, dest / "references", dirs_exist_ok=True)

        return dest

    def remove_skill(self, skill_name: str, target_dir: Path) -> bool:
        """Remove a skill directory from Claude's skills location."""
        skill_dir = target_dir / skill_name
        if skill_dir.is_dir():
            shutil.rmtree(skill_dir)
            return True
        return False

    def list_installed(self, target_dir: Path) -> list[str]:
        """List installed skills by scanning for SKILL.md files."""
        if not target_dir.is_dir():
            return []
        return sorted(
            d.name for d in target_dir.iterdir() if d.is_dir() and (d / "SKILL.md").exists()
        )

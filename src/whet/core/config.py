"""Global and local configuration model."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from whet.core.paths import bundled_dir

CONFIG_PATH = Path.home() / ".config" / "whet" / "config.json"


class Platform(str, Enum):
    """Supported AI coding agent platforms."""

    CLAUDE = "claude"
    ANTIGRAVITY = "antigravity"
    CURSOR = "cursor"
    COPILOT = "copilot"


class PlatformPaths(BaseModel):
    """Platform-specific paths for skill installation."""

    global_dir: Path
    local_dir: Path


PLATFORM_PATHS: dict[Platform, PlatformPaths] = {
    Platform.CLAUDE: PlatformPaths(
        global_dir=Path.home() / ".claude" / "skills",
        local_dir=Path(".claude") / "skills",
    ),
    Platform.ANTIGRAVITY: PlatformPaths(
        global_dir=Path.home() / ".gemini" / "antigravity" / "skills",
        local_dir=Path(".agent") / "skills",
    ),
    Platform.CURSOR: PlatformPaths(
        global_dir=Path.home() / ".cursor" / "rules",
        local_dir=Path(".cursor") / "rules",
    ),
    Platform.COPILOT: PlatformPaths(
        global_dir=Path.home() / ".github",
        local_dir=Path(".github"),
    ),
}


class WhetConfig(BaseModel):
    """Whet configuration persisted to disk."""

    target: Platform = Platform.CLAUDE
    skills_dir: Path = Field(default_factory=lambda: bundled_dir("skills"))
    archetypes_dir: Path = Field(default_factory=lambda: bundled_dir("archetypes"))

    @classmethod
    def load(cls) -> WhetConfig:
        """Load config from disk, falling back to defaults."""
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
            return cls(**data)
        return cls()

    def save(self) -> None:
        """Persist user-set values to disk."""
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {"target": self.target.value}
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")

    def get_platform_paths(self) -> PlatformPaths:
        """Get paths for the current target platform."""
        return PLATFORM_PATHS[self.target]

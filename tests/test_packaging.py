"""Guard that the built wheel actually ships the product.

whet's value is the skills, archetypes, and settings — not the CLI that copies them.
The wheel once packaged only `src/whet`, so `uv tool install whet` produced a CLI
that reported "No skills found" outside a source checkout. These tests assert the
bundled content is both declared for packaging and reachable at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib

from whet.core.paths import bundled_dir

BUNDLED = ("skills", "archetypes", "settings")


def test_wheel_force_includes_bundled_content() -> None:
    """Bundled dirs live outside src/whet, so the wheel needs an explicit include."""
    with open("pyproject.toml", "rb") as f:
        raw = tomllib.load(f)

    include = raw["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    missing = [name for name in BUNDLED if name not in include]
    assert not missing, (
        f"wheel does not force-include {missing}; an installed whet would find no skills"
    )
    for name in BUNDLED:
        assert include[name] == f"whet/_data/{name}", (
            f"{name} must map to whet/_data/{name} so bundled_dir() resolves it"
        )


def test_sdist_includes_bundled_content() -> None:
    with open("pyproject.toml", "rb") as f:
        raw = tomllib.load(f)

    include = raw["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    missing = [name for name in BUNDLED if f"{name}/" not in include]
    assert not missing, f"sdist does not include {missing}"


def test_sdist_does_not_ship_removed_directories() -> None:
    """agents/ was deleted; packaging must not still reference it."""
    with open("pyproject.toml", "rb") as f:
        raw = tomllib.load(f)

    include = raw["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    stale = [entry for entry in include if not Path(entry.rstrip("/")).exists()]
    assert not stale, f"packaging references directories that no longer exist: {stale}"


def test_bundled_dir_resolves_in_a_source_checkout() -> None:
    """Development must keep working with no build step."""
    for name in BUNDLED:
        resolved = bundled_dir(name)
        assert resolved.is_dir(), f"bundled_dir({name!r}) -> {resolved} does not exist"


def test_bundled_skills_are_the_real_library() -> None:
    """Catch a resolver that finds an empty or wrong directory."""
    skills = list(bundled_dir("skills").glob("*/SKILL.md"))
    assert len(skills) >= 25, f"bundled skills dir looks wrong: {len(skills)} SKILL.md files"

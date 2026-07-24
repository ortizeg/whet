"""Locate whet's bundled content (skills, archetypes, settings).

These directories live at the repository root during development, but must also be
found when whet is installed as a wheel — where the package lands in site-packages
and the repository layout no longer exists above it. The build force-includes them
under `whet/_data/`, so resolution prefers that and falls back to the repo layout.
"""

from __future__ import annotations

from pathlib import Path

# .../src/whet/core/paths.py -> parents[1] is the `whet` package.
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# .../src/whet/core/paths.py -> parents[3] is the repository root in a source checkout.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_BUNDLED = _PACKAGE_ROOT / "_data"


def bundled_dir(name: str) -> Path:
    """Path to a bundled content directory such as "skills" or "archetypes".

    Prefers the packaged copy so an installed wheel works outside any checkout;
    falls back to the repository root so a source checkout keeps working with no
    build step.
    """
    packaged = _BUNDLED / name
    if packaged.is_dir():
        return packaged
    return _REPO_ROOT / name

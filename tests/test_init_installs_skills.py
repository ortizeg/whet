"""`whet init` must install the archetype's skills, not just print them.

An archetype's value over a plain folder copy is the skill set it composes. While
init only printed a `whet add ...` hint, that value was left as homework and the
`[skills]` list was effectively advisory.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from whet.cli import app
from whet.core.config import PLATFORM_PATHS, Platform

runner = CliRunner()


def _skills_dir(project: Path) -> Path:
    return project / PLATFORM_PATHS[Platform.CLAUDE].local_dir


def test_init_installs_required_skills(tmp_path: Path) -> None:
    dest = tmp_path / "demo-svc"
    result = runner.invoke(
        app, ["init", "cv-inference-service", "--name", "demo-svc", "--output", str(dest)]
    )
    assert result.exit_code == 0, result.output

    installed = {d.name for d in _skills_dir(dest).iterdir() if d.is_dir()}
    # Declared required skills for this archetype must all be present.
    assert {"fastapi", "onnx", "pydantic", "loguru", "docker-cv", "testing"} <= installed


def test_init_installs_reference_files(tmp_path: Path) -> None:
    """Progressive-disclosure references must travel with the skill, or deep-dive links 404."""
    dest = tmp_path / "demo-svc"
    result = runner.invoke(
        app, ["init", "cv-inference-service", "--name", "demo-svc", "--output", str(dest)]
    )
    assert result.exit_code == 0, result.output

    refs = list((_skills_dir(dest) / "onnx" / "references").glob("*.md"))
    assert refs, "onnx installed without its references/ directory"


def test_init_no_skills_flag_scaffolds_only(tmp_path: Path) -> None:
    dest = tmp_path / "demo-svc"
    result = runner.invoke(
        app,
        [
            "init",
            "cv-inference-service",
            "--name",
            "demo-svc",
            "--output",
            str(dest),
            "--no-skills",
        ],
    )
    assert result.exit_code == 0, result.output

    assert (dest / "pyproject.toml").is_file(), "project should still be scaffolded"
    assert not _skills_dir(dest).exists(), "--no-skills must not install anything"


def test_init_with_recommended_installs_more(tmp_path: Path) -> None:
    base = tmp_path / "a"
    extra = tmp_path / "b"
    runner.invoke(app, ["init", "cv-inference-service", "--name", "a", "--output", str(base)])
    runner.invoke(
        app,
        [
            "init",
            "cv-inference-service",
            "--name",
            "b",
            "--output",
            str(extra),
            "--with-recommended",
        ],
    )

    n_base = len([d for d in _skills_dir(base).iterdir() if d.is_dir()])
    n_extra = len([d for d in _skills_dir(extra).iterdir() if d.is_dir()])
    assert n_extra > n_base, "--with-recommended should install additional skills"

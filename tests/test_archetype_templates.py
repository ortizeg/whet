"""Render every archetype and verify the generated project is actually usable.

The archetypes previously shipped templates that could not import their own entry
point, crashed on the first training step, and failed their own lint gate — none of
which any test noticed, because nothing ever rendered them. These tests do.

They run whet's real scaffolding code path (`render_template`), not a reimplementation,
so a regression in the engine is caught too.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib

from whet.scaffold.engine import Archetype, ScaffoldContext, discover_archetypes, render_template

ARCHETYPES_DIR = Path("archetypes")

# Modules that ship with Python; an import of these needs no declared dependency.
STDLIB = set(sys.stdlib_module_names)


def all_archetypes() -> list[Archetype]:
    return discover_archetypes(ARCHETYPES_DIR)


def _context() -> ScaffoldContext:
    return ScaffoldContext(
        project_name="Demo Proj",
        description="A demo project",
        author="Dev",
        python_version="3.11",
    )


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Render every archetype once and reuse the output across tests."""
    out: dict[str, Path] = {}
    for archetype in all_archetypes():
        target = tmp_path_factory.mktemp(f"render_{archetype.path.name.replace('-', '_')}")
        render_template(archetype, target, _context())
        out[archetype.path.name] = target
    return out


def _text_files(root: Path) -> list[Path]:
    suffixes = {".py", ".toml", ".yaml", ".yml", ".md", ".txt", ".cfg", ".ipynb"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes]


@pytest.mark.parametrize("archetype", all_archetypes(), ids=lambda a: a.path.name)
def test_render_leaves_no_unsubstituted_variables(
    archetype: Archetype, rendered: dict[str, Path]
) -> None:
    """`{{var}}` is inert in this engine, and a stray `${var}` means a typo'd name."""
    root = rendered[archetype.path.name]
    offenders: list[str] = []

    for path in _text_files(root):
        content = path.read_text()
        rel = path.relative_to(root)
        if "{{" in content:
            offenders.append(f"{rel}: contains inert '{{{{' placeholder")
        # Hydra-style ${a.b} interpolation is legitimate; a bare ${word} is not.
        for token in ("${project", "${package", "${description", "${author", "${python_version"):
            if token in content:
                offenders.append(f"{rel}: unsubstituted {token}")

    joined = "\n".join(offenders)
    assert not offenders, f"{archetype.path.name} rendered with placeholders:\n{joined}"


@pytest.mark.parametrize("archetype", all_archetypes(), ids=lambda a: a.path.name)
def test_rendered_python_is_syntactically_valid(
    archetype: Archetype, rendered: dict[str, Path]
) -> None:
    """A rendered project must at minimum parse."""
    root = rendered[archetype.path.name]
    for path in root.rglob("*.py"):
        source = path.read_text()
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - only on failure
            rel = path.relative_to(root)
            pytest.fail(f"{archetype.path.name}: {rel} is not valid Python: {exc}")


@pytest.mark.parametrize("archetype", all_archetypes(), ids=lambda a: a.path.name)
def test_rendered_project_passes_ruff(archetype: Archetype, rendered: dict[str, Path]) -> None:
    """Every archetype documents `ruff check .` — so it had better pass it."""
    root = rendered[archetype.path.name]
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{archetype.path.name} fails its own ruff gate:\n{result.stdout}\n{result.stderr}"
    )


def _declared_dependencies(root: Path) -> set[str]:
    """Top-level distribution names declared in the rendered pyproject.toml."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return set()

    with open(pyproject, "rb") as f:
        raw = tomllib.load(f)

    project = raw.get("project", {})
    specs: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        specs.extend(extra)
    for group in raw.get("dependency-groups", {}).values():
        specs.extend(g for g in group if isinstance(g, str))

    names: set[str] = set()
    for spec in specs:
        head = spec.split(";")[0].strip()
        for sep in ("[", ">", "<", "=", "!", "~", " "):
            head = head.split(sep)[0]
        if head:
            names.add(head.strip().lower().replace("-", "_"))
    return names


# Import name -> distribution name, where they differ.
IMPORT_ALIASES = {
    "yaml": "pyyaml",
    "cv2": "opencv_python",
    "PIL": "pillow",
    "sklearn": "scikit_learn",
    "lightning": "lightning",
    "hydra": "hydra_core",
    "onnxruntime": "onnxruntime",
    "dotenv": "python_dotenv",
    "multipart": "python_multipart",
    "nbformat": "nbformat",
}


@pytest.mark.parametrize("archetype", all_archetypes(), ids=lambda a: a.path.name)
def test_rendered_imports_are_declared(archetype: Archetype, rendered: dict[str, Path]) -> None:
    """Every third-party import must be a declared dependency.

    This is the check that would have caught the pytorch template shipping zero
    `[project].dependencies` while importing torch, lightning, and loguru.
    """
    root = rendered[archetype.path.name]
    declared = _declared_dependencies(root)
    local_pkgs = (
        {p.name for p in (root / "src").iterdir() if p.is_dir()}
        if (root / "src").is_dir()
        else set()
    )

    missing: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue

            for name in roots:
                if not name or name in STDLIB or name in local_pkgs:
                    continue
                dist = IMPORT_ALIASES.get(name, name).lower().replace("-", "_")
                if dist not in declared:
                    missing.add(f"{name} (expected dependency '{dist}')")

    assert not missing, (
        f"{archetype.path.name} imports undeclared packages: {sorted(missing)}\n"
        f"declared: {sorted(declared)}"
    )


@pytest.mark.parametrize("archetype", all_archetypes(), ids=lambda a: a.path.name)
def test_rendered_project_has_src_layout_and_tests(
    archetype: Archetype, rendered: dict[str, Path]
) -> None:
    """src-layout is mandatory across archetypes, and a project needs tests."""
    root = rendered[archetype.path.name]
    assert (root / "pyproject.toml").is_file(), f"{archetype.path.name}: no pyproject.toml"
    assert (root / "src").is_dir(), f"{archetype.path.name}: not src-layout"
    assert (root / "tests").is_dir(), f"{archetype.path.name}: no tests/ directory"


def test_notebooks_are_substitutable() -> None:
    """`.ipynb` must be treated as text so `${package_name}` renders inside notebooks.

    Notebooks are JSON, so substitution is safe. When they were excluded, a template
    notebook importing from `${package_name}` shipped the literal placeholder — and
    then tripped test_render_leaves_no_unsubstituted_variables with a confusing
    failure pointing at the notebook rather than at the engine.
    """
    from whet.scaffold.engine import _is_text_file

    assert _is_text_file(Path("notebooks/01-explore.ipynb")), (
        ".ipynb must be substitutable; otherwise notebooks ship literal ${...} placeholders"
    )

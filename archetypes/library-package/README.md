# Library Package Archetype

A project template for a **publishable, pip-installable Python library**. It
generates a complete, working src-layout package: hatchling build backend, a
typed public API with a `py.typed` marker, a console-script entry point, a real
test suite, MkDocs documentation, MIT license, a Keep a Changelog file, and CI
workflows for lint/typecheck/test, docs deployment, and PyPI publishing.

The generated project passes its own `ruff check .`, `ruff format --check .`,
`mypy src/ --strict`, and `pytest` gates from the moment it is created, and
`python -m build` produces a wheel that installs and runs.

## Purpose

Teams accumulate shared utilities, model wrappers, data helpers, and evaluation
code that get copy-pasted between projects. This archetype turns that ad-hoc
sharing into a properly packaged, versioned, documented library.

Its reason to exist over "just add the `pypi` skill" is the **packaging
furniture**: the license, the changelog, the `py.typed` marker, the release
workflow, the docs site, and the version single-sourcing are all wired up and
verified, not described.

## Use Cases

- **Shared model architectures** — custom backbones, detection heads, or
  segmentation decoders used across multiple training projects.
- **Data processing utilities** — image loading, format conversion, annotation
  parsing, augmentation pipeline builders.
- **Evaluation toolkits** — custom metric implementations and benchmark runners
  shared across experiments.
- **Training utilities** — schedulers, optimizers, loss functions, callbacks.
- **Inference wrappers** — model loading, preprocessing, and postprocessing
  behind a clean prediction API.

The archetype deliberately ships **no** CV/ML dependencies. Add `numpy`, `torch`
or anything else with `pixi add`, and mirror it into `[project.dependencies]`.

## Generated Structure

```
${project_slug}/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # ruff + mypy + pytest (two Python versions) + build
│       ├── docs.yml                # mkdocs build --strict, gh-deploy on main
│       └── publish.yml             # build + twine check + PyPI trusted publishing
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md                    # Keep a Changelog format
├── LICENSE.txt                     # MIT, with the author substituted in
├── README.md
├── mkdocs.yml                      # Material theme + mkdocstrings
├── pixi.toml                       # canonical dev environment and tasks
├── pyproject.toml                  # hatchling, deps, ruff/mypy/pytest/coverage config
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   └── api.md                      # mkdocstrings API reference
├── examples/
│   └── basic_usage.py
├── src/${package_name}/
│   ├── __init__.py                 # the public API + __version__
│   ├── cli.py                      # console-script entry point
│   ├── config.py                   # LibraryConfig (Pydantic V2, frozen)
│   ├── core.py                     # Registry / RegistryError
│   └── py.typed                    # PEP 561 marker
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_cli.py
    ├── test_config.py
    └── test_core.py
```

## What the Generated Library Does

The starting API is small and real, not a stub — it is meant to be extended or
replaced, but it works and is fully tested.

- **`Registry[T]`** — a typed, name-keyed registry of factories. It turns a
  configuration string into an object without the call site importing the
  concrete implementation. Keys are case- and whitespace-insensitive; duplicate
  registration raises instead of silently shadowing.
- **`RegistryError`** — subclasses `KeyError`, so existing dictionary-style
  handlers keep working. Lookup failures list the known keys.
- **`component_registry`** — a package-level `Registry` for plugins to populate.
- **`LibraryConfig`** — a frozen Pydantic V2 model with `extra="forbid"` and a
  `from_env()` loader that reads `ENV_PREFIX`-scoped environment variables.
- **A console script** — `${project_slug} info` and `${project_slug} components`
  emit JSON on stdout and Loguru diagnostics on stderr.

```python
from ${package_name} import LibraryConfig, Registry

encoders: Registry[str] = Registry("encoders")


@encoders.register("upper")
def upper_encoder(text: str) -> str:
    return text.upper()


encoders.create("UPPER", "hi")  # "HI"

config = LibraryConfig.from_env()
config.max_items  # 1024 unless overridden in the environment
```

## Key Features

- **src-layout packaging** so the installed package is what gets tested, not the
  working directory.
- **`pyproject.toml` as the single source of truth** for metadata, dependencies,
  and the ruff / mypy / pytest / coverage configuration.
- **Version single-sourcing** — `[tool.hatch.version]` reads `__version__` from
  `src/${package_name}/__init__.py`, so the version string exists in one place.
- **Full type safety** — `mypy --strict` over `src/` and `tests/`, plus the
  `py.typed` marker so downstream consumers get the shipped hints.
- **MkDocs Material documentation** with a mkdocstrings API reference generated
  from Google-style docstrings, built with `--strict` in CI.
- **Loguru logging with no sinks installed by the library** — an imported
  library has no business rerouting an application's logs. Only `cli.py` adds a
  sink, and only when the CLI actually runs.
- **Automated PyPI publishing** on a published GitHub Release, via trusted
  publishing (no stored API token).

## Configuration Variables

`whet init` substitutes these six variables into both file contents and
file/directory names. There are no others.

| Variable | Description | Default |
|---|---|---|
| `${project_name}` | Human-readable library name | Required |
| `${project_slug}` | PyPI distribution name (hyphenated) | Derived from `project_name` |
| `${package_name}` | Python import name (underscored) | Derived from `project_slug` |
| `${description}` | One-line package description | Empty |
| `${author}` | Author or organization name | Empty |
| `${python_version}` | Minimum Python version | `3.11` |

Author email, license choice, and documentation URL are **not** variables — edit
`pyproject.toml`, `LICENSE.txt`, and `mkdocs.yml` after generating.

## Dependencies

Runtime dependencies are kept minimal, because every entry is imposed on every
downstream consumer.

```toml
[project]
dependencies = [
    "loguru>=0.7",
    "pydantic>=2.6",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.8", "mypy>=1.11",
       "pre-commit>=3.5", "build>=1.2", "twine>=5.1"]
docs = ["mkdocs>=1.6", "mkdocs-material>=9.5", "mkdocstrings[python]>=0.26"]
all = ["${project_slug}[dev,docs]"]
```

## Usage

### Generate

```bash
whet init library-package --name "My Library"
```

### Develop

Pixi is the canonical environment manager for generated projects.

```bash
pixi install                  # create the environment
pixi add <name>               # add a conda dependency
pixi add --pypi <name>        # add a PyPI dependency
pre-commit install            # install the git hooks
```

Inside the environment (`pixi shell`, or via `pixi run <task>`), the commands are
tool-agnostic:

```bash
pytest                        # test suite
pytest --cov                  # with coverage
ruff check .                  # lint
ruff format .                 # format
mypy src/ --strict            # type check
```

Pixi tasks wrap the same commands: `pixi run test`, `pixi run lint`,
`pixi run typecheck`, `pixi run quality`.

### Document

```bash
pixi run -e docs docs-serve   # live reload
pixi run -e docs docs-build   # strict build into site/
```

### Release

1. Move the `[Unreleased]` entries in `CHANGELOG.md` under a new version heading.
2. Bump `__version__` in `src/${package_name}/__init__.py`.
3. Tag and push:

```bash
git tag v0.2.0
git push origin v0.2.0
```

4. Publish the GitHub Release. `publish.yml` builds, runs `twine check`, and
   uploads to PyPI.

To build and upload by hand:

```bash
python -m build
twine check dist/*
twine upload --repository testpypi dist/*   # staging
twine upload dist/*                         # production
```

## API Design Principles

### Minimal Public API

Export only what users need from `__init__.py`. Everything else is an
implementation detail that can be refactored without a major version bump. The
generated `CHANGELOG.md` states this contract explicitly.

### Registry Pattern

Use a registry for extensible components. Callers instantiate by name, which
keeps configuration-driven code decoupled from import paths.

```python
from ${package_name} import Registry

models: Registry[object] = Registry("models")


@models.register("my_custom_net")
class MyCustomNet:
    def __init__(self, num_classes: int) -> None: ...


model = models.create("my_custom_net", num_classes=10)
```

### Version Policy

Semantic versioning: patch for fixes, minor for backward-compatible additions,
major for breaking changes. The public API is what `__init__.py` exports.

## Customization Guide

### Adding a Module

1. Add `src/${package_name}/<module>.py` with Google-style docstrings.
2. Re-export the public names from `__init__.py` and add them to `__all__`.
3. Add tests in `tests/test_<module>.py`.
4. Add the module to `docs/api.md` as a mkdocstrings block.
5. Record the addition under `[Unreleased]` in `CHANGELOG.md`.

### Optional Dependencies

For features needing heavy dependencies (ONNX export, a specific vision
backend), add an extra in `[project.optional-dependencies]` and guard the import
with a `try`/`except ImportError` that names the extra to install.

### Namespace Packages

To split a library across repositories (`mylib.core`, `mylib.contrib`), use
implicit namespace packages by omitting `__init__.py` at the namespace level and
pointing `[tool.hatch.build.targets.wheel].packages` at the subpackage.

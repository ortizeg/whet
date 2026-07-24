# Library Package

Reusable Python package designed for PyPI distribution with proper documentation, testing, and versioning.

## Purpose

This archetype creates a well-structured Python package that can be published to PyPI. It includes proper package metadata, comprehensive testing, API documentation with MkDocs, and a GitHub Actions release workflow. Use this when building reusable tools, utilities, or model libraries that others will install via pip.

## Directory Structure

```
${project_slug}/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── docs.yml
│       └── publish.yml
├── docs/
│   ├── api.md
│   ├── getting-started.md
│   └── index.md
├── examples/
│   └── basic_usage.py
├── src/
│   └── ${package_name}/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── core.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_config.py
│   └── test_core.py
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── LICENSE.txt
├── README.md
├── mkdocs.yml
├── pixi.toml
└── pyproject.toml
```

## Key Features

- **PEP 561 compliant** -- `py.typed` marker for downstream type checking
- **src layout** -- proper package isolation during testing
- **API documentation** -- MkDocs with mkdocstrings for auto-generated API docs
- **Release workflow** -- automated PyPI publishing on GitHub release tags
- **Semantic versioning** -- version managed in `pyproject.toml`

## Usage

```bash
# Install in development mode
pixi install

# Run tests
pytest

# Build package
python -m build

# Publish to PyPI (via GitHub Actions on tag)
git tag v0.1.0 && git push --tags
```

## Customization

- Define public API in `src/{{package_name}}/__init__.py`
- Add modules under `core/` or `utils/`
- Configure optional dependencies in `pyproject.toml` extras
- Extend documentation in `docs/guides/`

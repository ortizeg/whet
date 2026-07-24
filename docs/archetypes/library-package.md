# Library Package

Reusable Python package designed for PyPI distribution with proper documentation, testing, and versioning.

## Purpose

This archetype creates a well-structured Python package that can be published to PyPI. It includes proper package metadata, comprehensive testing, API documentation with MkDocs, and a GitHub Actions release workflow. Use this when building reusable tools, utilities, or model libraries that others will install via pip.

## Directory Structure

```
{{project_slug}}/
├── src/{{package_name}}/
│   ├── __init__.py            # Public API exports
│   ├── py.typed               # PEP 561 type marker
│   ├── core/
│   │   ├── __init__.py
│   │   └── module.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── types.py               # Public type definitions
├── docs/
│   ├── mkdocs.yml
│   ├── index.md
│   ├── api/                   # Auto-generated API docs
│   └── guides/
├── tests/
│   ├── conftest.py
│   ├── test_core.py
│   └── test_utils.py
├── .github/workflows/
│   ├── test.yml
│   ├── docs.yml
│   └── release.yml            # PyPI publishing
└── ...
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

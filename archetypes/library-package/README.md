# Library Package Archetype

A comprehensive project template for building reusable Python packages for computer vision and machine learning. This archetype provides the full infrastructure needed to develop, test, document, and publish a library to PyPI, including src-layout packaging, comprehensive test suites, automated documentation, and CI/CD pipelines for versioned releases.

## Purpose

Many CV/ML teams accumulate shared utilities, custom model architectures, data processing functions, and evaluation tools that are copy-pasted between projects. The Library Package archetype transforms this ad-hoc code sharing into a properly packaged, versioned, and documented library that can be installed via `pip install` and imported cleanly across any project in the organization.

This archetype enforces modern Python packaging best practices: src-layout for clean import isolation, `pyproject.toml` as the single source of packaging metadata, strict type annotations with `py.typed` marker for downstream type checking, and semantic versioning managed through git tags. The included CI/CD pipeline automates testing across multiple Python versions, builds documentation, and publishes releases to PyPI on tagged commits.

A well-packaged library reduces duplicated code, ensures consistent behavior across projects, enables proper dependency management through version pinning, and creates a foundation for team-wide code review and quality standards.

## Use Cases

- **Shared model architectures** -- Package custom backbone networks, detection heads, or segmentation decoders that are used across multiple training projects.
- **Data processing utilities** -- Consolidate image loading, format conversion, annotation parsing, and augmentation pipeline builders into a reusable library.
- **Evaluation toolkits** -- Package COCO evaluation, custom metric implementations, and benchmark runners for consistent evaluation across experiments.
- **Training utilities** -- Share learning rate schedulers, custom optimizers, loss functions, and callback implementations.
- **Dataset interfaces** -- Create standardized dataset classes for proprietary or domain-specific data formats.
- **Inference wrappers** -- Package model loading, preprocessing, and postprocessing logic into clean prediction APIs.

## Directory Structure

```
{{project_slug}}/
├── .github/
│   └── workflows/
│       ├── test.yml                    # Test across Python versions
│       ├── docs.yml                    # Build and deploy docs
│       ├── publish.yml                 # Publish to PyPI on release
│       └── ci.yml                     # Lint, type check, tests
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml                      # Package metadata and build config
├── README.md
├── LICENSE
├── CHANGELOG.md                        # Version history
├── src/{{package_name}}/
│   ├── __init__.py                     # Public API and __version__
│   ├── py.typed                        # PEP 561 type marker
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbones.py               # Backbone architectures
│   │   ├── heads.py                    # Task-specific heads
│   │   └── registry.py                # Model registry pattern
│   ├── data/
│   │   ├── __init__.py
│   │   ├── transforms.py             # Transform pipelines
│   │   ├── loaders.py                 # Dataset loaders
│   │   └── io.py                      # Format read/write
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── detection.py               # Detection metrics (mAP)
│   │   ├── segmentation.py           # Segmentation metrics (IoU)
│   │   └── classification.py         # Classification metrics
│   ├── losses/
│   │   ├── __init__.py
│   │   └── focal.py                   # Custom loss functions
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py           # Drawing and plotting
│       └── io.py                      # General I/O helpers
├── docs/
│   ├── mkdocs.yml                     # MkDocs configuration
│   ├── index.md                       # Documentation home
│   ├── getting-started.md
│   ├── api/                           # Auto-generated API reference
│   │   └── .gitkeep
│   ├── guides/
│   │   ├── models.md
│   │   ├── data.md
│   │   └── metrics.md
│   └── changelog.md                   # Symlink to CHANGELOG.md
├── examples/
│   ├── basic_usage.py                 # Minimal usage example
│   ├── custom_model.py                # Model customization example
│   └── evaluation.py                  # Evaluation pipeline example
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures and test data
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models.py
│   │   ├── test_transforms.py
│   │   ├── test_metrics.py
│   │   └── test_losses.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_pipelines.py
│   └── fixtures/
│       └── sample_data/               # Test images and annotations
│           └── .gitkeep
└── benchmarks/
    ├── __init__.py
    └── bench_transforms.py            # Performance benchmarks
```

## Key Features

- **Src-layout packaging** that prevents accidental imports from the source tree and ensures the installed package is tested, not the local directory.
- **pyproject.toml** as the single source of truth for package metadata, dependencies, build system, and tool configuration.
- **Semantic versioning** managed through git tags and dynamic version resolution, eliminating version string duplication.
- **Full type safety** with strict mypy configuration and the `py.typed` PEP 561 marker so downstream consumers get type checking support.
- **MkDocs documentation** with Material theme, auto-generated API reference from docstrings, and GitHub Pages deployment.
- **Multi-version CI testing** across Python 3.11 and 3.12 with comprehensive unit and integration test suites.
- **Automated PyPI publishing** triggered by git tag pushes with proper build isolation and artifact verification.
- **Model registry pattern** for registering and discovering model architectures by name string.

## Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `{{project_name}}` | Human-readable library name | Required |
| `{{project_slug}}` | PyPI package name (hyphenated) | Auto-generated |
| `{{package_name}}` | Python import name (underscored) | Auto-generated |
| `{{author_name}}` | Author or organization name | Required |
| `{{email}}` | Author or organization email | Required |
| `{{description}}` | One-line package description for PyPI | Required |
| `{{version}}` | Initial version | 0.1.0 |
| `{{python_version}}` | Minimum Python version | 3.11 |
| `{{license}}` | License type (MIT, Apache-2.0, BSD-3) | MIT |
| `{{docs_url}}` | Documentation URL | Auto-generated |

## Dependencies

Core dependencies are kept minimal to avoid constraining downstream consumers.

```toml
[project]
dependencies = [
    "numpy>=1.26",
    "pillow>=10.0",
    "torch>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "mypy>=1.7",
    "ruff>=0.1",
    "pre-commit>=3.5",
]
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.4",
    "mkdocstrings[python]>=0.24",
]
all = ["{{package_name}}[dev,docs]"]
```

## Usage

### Development Setup

```bash
# Install dependencies including dev extras
pixi install

# Install the package in editable mode
pixi run pip install -e ".[dev]"

# Run the test suite
pixi run pytest

# Run type checking
pixi run mypy src/

# Run linting
pixi run ruff check src/ tests/
```

### Using the Library

```python
import {{package_name}}

# Use the model registry
from {{package_name}}.models import create_model
model = create_model("resnet50", pretrained=True, num_classes=10)

# Use data transforms
from {{package_name}}.data import create_transform_pipeline
transform = create_transform_pipeline("train", input_size=224)

# Use evaluation metrics
from {{package_name}}.metrics import MeanAveragePrecision
metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
metric.update(predictions, targets)
result = metric.compute()
```

### Building Documentation

```bash
# Serve docs locally with hot reload
pixi run mkdocs serve

# Build static documentation site
pixi run mkdocs build

# Documentation is auto-deployed to GitHub Pages on push to main
```

### Publishing a Release

```bash
# Update CHANGELOG.md with release notes

# Tag a release (triggers CI publish)
git tag v0.2.0
git push origin v0.2.0

# The CI pipeline will:
# 1. Run full test suite
# 2. Build source and wheel distributions
# 3. Publish to PyPI
# 4. Create a GitHub Release with changelog notes
```

## API Design Principles

### Minimal Public API

Export only what users need from `__init__.py`. Internal implementation details should use underscore-prefixed names or live in submodules not exposed at the top level. This allows refactoring internals without breaking downstream code.

### Registry Pattern

Use the registry pattern for extensible components like model architectures. Users register new implementations by name and instantiate them through a factory function, enabling configuration-driven code without import coupling.

```python
from {{package_name}}.models.registry import register_model, create_model

@register_model("my_custom_net")
class MyCustomNet(nn.Module):
    ...

model = create_model("my_custom_net", num_classes=10)
```

### Version Policy

Follow semantic versioning strictly. Patch versions for bug fixes, minor versions for backward-compatible additions, major versions for breaking changes. The public API is defined by what is exported from top-level `__init__.py` files.

## Customization Guide

### Adding a New Module

1. Create the module in the appropriate `src/{{package_name}}/` subdirectory.
2. Add comprehensive docstrings in Google style for MkDocs auto-generation.
3. Export the public API from the subdirectory's `__init__.py`.
4. Write unit tests in `tests/unit/` with edge cases and type validation.
5. Add a usage example in `examples/`.
6. Update the relevant documentation guide in `docs/guides/`.

### Optional Dependencies

For features that require heavy dependencies (e.g., ONNX export, specific vision backends), use optional dependency groups in `pyproject.toml` and guard imports with try/except blocks that raise informative `ImportError` messages pointing users to the correct install extra.

### Namespace Packages

If the library needs to be split across multiple repositories (e.g., `mylib.core` and `mylib.contrib`), configure namespace packages using the implicit namespace package approach by omitting `__init__.py` at the namespace level.

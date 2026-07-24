---
name: pypi
description: >
  Use this skill when packaging and publishing a Python project to PyPI — configuring
  pyproject.toml build metadata, src layout, version management, the hatchling or
  setuptools build backend, entry points, trusted publishing via GitHub Actions, and
  TestPyPI staging. Reach for it any time you'd otherwise figure out how to build a wheel
  and upload a release, even if the user just says "publish this package" or "make it
  pip-installable". For managing the local dev environment and dependencies, see pixi.
---

# PyPI Publishing Skill

Packaging and publishing Python projects to PyPI: `pyproject.toml` configuration, src layout, version management, building, TestPyPI staging, trusted publishers via GitHub Actions, and release automation.

## Project Layout

Use the `src` layout — it prevents accidental imports from the working directory (a common "works on my machine" bug) and is the recommended standard.

```
my-cv-package/
├── src/my_cv_package/
│   ├── __init__.py
│   ├── py.typed              # PEP 561 marker
│   ├── models/               # detector.py, classifier.py
│   ├── data/                 # transforms.py
│   ├── utils/                # io.py, visualization.py
│   └── cli.py
├── tests/
├── pyproject.toml
├── LICENSE
└── README.md
```

## pyproject.toml Configuration

### Hatchling (recommended)

Modern, fast build backend with good defaults.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-cv-package"
version = "0.1.0"
description = "Computer vision utilities for object detection and classification"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Your Name", email = "you@example.com"}]
keywords = ["computer-vision", "deep-learning", "object-detection"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = [
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "torch>=2.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.4", "mypy>=1.0", "pre-commit>=3.0"]
docs = ["mkdocs>=1.5", "mkdocs-material>=9.0", "mkdocstrings[python]>=0.24"]
all = ["my-cv-package[dev,docs]"]

[project.urls]
Homepage = "https://github.com/yourname/my-cv-package"
Documentation = "https://yourname.github.io/my-cv-package"
Repository = "https://github.com/yourname/my-cv-package"
Issues = "https://github.com/yourname/my-cv-package/issues"

[project.scripts]
my-cv-tool = "my_cv_package.cli:main"
```

### Setuptools (alternative)

Swap the build backend and declare package discovery. Combine with `setuptools-scm` for git-tag versioning:

```toml
[build-system]
requires = ["setuptools>=68.0", "setuptools-scm>=8.0"]
build-backend = "setuptools.build_meta"

[project]
name = "my-cv-package"
dynamic = ["version"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools_scm]
write_to = "src/my_cv_package/_version.py"
```

## Version Management

Three approaches — pick one:

**Manual** — keep `version = "0.1.0"` in `pyproject.toml`; read it at runtime from installed metadata so `__init__.py` never drifts:

```python
# src/my_cv_package/__init__.py
from importlib.metadata import version

__version__ = version("my-cv-package")
```

**Git-tag derived (setuptools-scm)** — set `dynamic = ["version"]` plus `[tool.setuptools_scm]` (see above), then tag:

```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

**Git-tag derived (hatch-vcs)** — the Hatchling equivalent:

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
dynamic = ["version"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/my_cv_package/_version.py"
```

## Entry Points

**CLI scripts** — installed as commands on the user's PATH:

```toml
[project.scripts]
my-cv-detect = "my_cv_package.cli:detect_main"
my-cv-train = "my_cv_package.cli:train_main"
```

```python
# src/my_cv_package/cli.py
import argparse


def detect_main() -> None:
    parser = argparse.ArgumentParser(description="Run object detection")
    parser.add_argument("input", help="Input image or video path")
    parser.add_argument("--model", default="yolov8n")
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    from my_cv_package.models.detector import detect
    detect(args.input, model=args.model, confidence=args.confidence)
```

**Plugin entry points** — let third parties register implementations you discover at runtime:

```toml
[project.entry-points."my_cv_package.models"]
resnet = "my_cv_package.models.classifier:ResNetClassifier"
yolo = "my_cv_package.models.detector:YOLODetector"
```

```python
from importlib.metadata import entry_points


def get_available_models() -> dict[str, type]:
    eps = entry_points(group="my_cv_package.models")
    return {ep.name: ep.load() for ep in eps}
```

## Building and Publishing

```bash
python -m build          # produces dist/*.tar.gz (sdist) + dist/*.whl (wheel)
unzip -l dist/*.whl      # inspect wheel contents

# Stage on TestPyPI first, then install to verify
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ my-cv-package

# Publish to the real index (API token via -u __token__ -p, or trusted publishing below)
twine upload dist/*
```

## Trusted Publisher (GitHub Actions)

Trusted publishers use OIDC to authenticate GitHub Actions to PyPI — no API tokens to manage. Register the repo/workflow/environment under your PyPI project settings, then use a tag-triggered workflow. Build once, fan out to test-install, TestPyPI, then PyPI:

```yaml
# .github/workflows/release.yml
name: Release to PyPI
on:
  push:
    tags: ["v*"]
permissions:
  contents: write
  id-token: write   # required for trusted publishing
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # for setuptools-scm
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build && python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  test-install:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12", "3.13"]
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - run: pip install dist/*.whl
      - run: python -c "import my_cv_package; print(my_cv_package.__version__)"

  publish-pypi:
    needs: test-install
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Add a `publish-testpypi` job (same shape, `repository-url: https://test.pypi.org/legacy/`, `environment: testpypi`) before `publish-pypi` to stage, and a `github-release` job using `softprops/action-gh-release@v2` with `generate_release_notes: true` after it.

## Semantic Release

Python Semantic Release automates version bumps, changelogs, and publishing from commit messages — no manual versioning.

Uses [Conventional Commits](https://www.conventionalcommits.org/): `feat:` bumps minor, `fix:` bumps patch, `feat!:` (or a `BREAKING CHANGE` footer) bumps major. `docs`, `chore`, `ci`, `refactor`, `test`, `style` do **not** trigger a release.

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
build_command = "pip install build && python -m build"
upload_to_pypi = true
upload_to_release = true
commit_message = "chore(release): v{version} [skip ci]"   # [skip ci] avoids CI loops
tag_format = "v{version}"

[tool.semantic_release.changelog]
changelog_file = "CHANGELOG.md"

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"          # cuts 0.3.0-dev.1 prereleases
```

Notes: `version_toml` replaces the deprecated `version_variable`; `build_command` runs before publishing (swap in `flit build`, `uv build`, etc. for other backends). See the **GitHub Actions** skill for the full CI workflow that runs semantic release on push to `main`/`develop` and publishes via trusted publishers.

## Release Checklist (manual alternative)

If not using semantic release:

```bash
pytest                                          # 1. tests pass
# 2. bump version in pyproject.toml (if not using scm) + update CHANGELOG.md
git commit -am "Release v0.2.0"                 # 3. commit
git tag -a v0.2.0 -m "Release version 0.2.0"    # 4. annotated tag
git push origin main && git push origin v0.2.0  # 5. push — Actions builds + publishes
pip install --upgrade my-cv-package             # 6. verify
```

## Including Package Data

Ship non-Python files (configs, default params) and load them via `importlib.resources`:

```toml
# hatchling
[tool.hatch.build.targets.wheel.force-include]
"configs" = "my_cv_package/configs"

# setuptools
[tool.setuptools.package-data]
my_cv_package = ["configs/*.yaml", "configs/*.json"]
```

```python
from importlib import resources
import yaml


def get_default_config() -> dict:
    cfg = resources.files("my_cv_package.configs").joinpath("default.yaml")
    with resources.as_file(cfg) as path:
        return yaml.safe_load(path.read_text())
```

## Best Practices

1. **Use src layout** — prevents accidental local imports during development.
2. **Use trusted publishers** — no API tokens to manage or leak.
3. **Test on TestPyPI first** — verify the package installs before hitting the real index.
4. **Declare all dependencies** — never assume packages are pre-installed.
5. **Use `opencv-python-headless`** — avoids GUI dependency conflicts on servers.
6. **Use optional dependencies** — put dev/docs/heavy extras in `[project.optional-dependencies]`.
7. **Automate releases** — tag-triggered workflows make publishing a single `git tag`.
8. **Include `py.typed`** — signals PEP 561 type support.
9. **Version with git tags** — setuptools-scm or hatch-vcs eliminate manual bumps.

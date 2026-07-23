# pixi.toml Manifest Reference

Scope: the structure of a pixi manifest — minimal and complete `pixi.toml` examples, and
the equivalent configuration embedded in `pyproject.toml`.

## Contents

- [What Is Pixi and Why Use It](#what-is-pixi-and-why-use-it)
- [Minimal pixi.toml](#minimal-pixitoml)
- [Complete pixi.toml for a Training Project](#complete-pixitoml-for-a-training-project)
- [Integration with pyproject.toml](#integration-with-pyprojecttoml)

## What Is Pixi and Why Use It

Pixi is a fast, cross-platform package manager built on the conda ecosystem. It replaces
conda, mamba, pip, virtualenv, Make, and requirements.txt files with a single `pixi.toml`
configuration file. It provides:

- **Reproducible environments**: Lock files ensure identical environments across machines.
- **Cross-platform support**: Works on Linux, macOS (Intel and Apple Silicon), and Windows.
- **Conda + PyPI integration**: Access packages from conda-forge, pytorch, and PyPI in the same environment.
- **Task runner**: Define project tasks (test, lint, train) directly in pixi.toml, replacing Makefiles.
- **Feature groups**: Organize optional dependencies into named features (dev, cuda, docs).
- **Fast resolution**: Uses the rattler solver for sub-second dependency resolution.

Every project in this framework uses pixi as the sole environment and dependency manager.
Never use conda, pip, or virtualenv directly.

## Minimal pixi.toml

```toml
[project]
name = "my-cv-project"
version = "0.1.0"
description = "A computer vision project"
authors = ["Enrique G. Ortiz <ortizeg@gmail.com>"]
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
```

## Complete pixi.toml for a Training Project

```toml
[project]
name = "semantic-segmentation"
version = "0.1.0"
description = "Semantic segmentation training pipeline"
authors = ["Enrique G. Ortiz <ortizeg@gmail.com>"]
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"
pillow = ">=10.0"

[pypi-dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
lightning = ">=2.2"
albumentations = ">=1.3"
pydantic = ">=2.6"
wandb = ">=0.16"

# -----------------------------------------------------------
# Feature groups: optional dependency sets
# -----------------------------------------------------------

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[feature.dev.pypi-dependencies]
pytest-mock = ">=3.12"

[feature.docs.dependencies]
mkdocs = ">=1.5"
mkdocstrings = { version = ">=0.24", extras = ["python"] }

[feature.onnx.pypi-dependencies]
onnx = ">=1.15"
onnxruntime = ">=1.17"

# -----------------------------------------------------------
# Environments: combine features into installable sets
# -----------------------------------------------------------

[environments]
default = { features = ["dev"], solve-group = "default" }
docs = { features = ["dev", "docs"], solve-group = "default" }
export = { features = ["dev", "onnx"], solve-group = "default" }

# -----------------------------------------------------------
# Tasks: commands you run with `pixi run <task>`
# -----------------------------------------------------------

[tasks]
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src --cov-report=term --cov-report=html --cov-fail-under=80"
lint = "ruff check ."
lint-fix = "ruff check . --fix"
format = "ruff format ."
format-check = "ruff format . --check"
typecheck = "mypy src/ --strict"
quality = { depends-on = ["lint", "format-check", "typecheck"] }
train = "python -m semantic_segmentation.train"
```

## Integration with pyproject.toml

Pixi can read project metadata from pyproject.toml, allowing the package definition and
the environment definition to coexist.

```toml
# pyproject.toml
[project]
name = "my-cv-project"
version = "0.1.0"
description = "A computer vision project"
requires-python = ">=3.11"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pixi.project]
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[tool.pixi.dependencies]
python = ">=3.11"
numpy = ">=1.26"

[tool.pixi.pypi-dependencies]
torch = ">=2.2"

[tool.pixi.feature.dev.dependencies]
pytest = ">=7.4"
ruff = ">=0.8"

[tool.pixi.tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
```

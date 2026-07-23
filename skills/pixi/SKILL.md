---
name: pixi
description: >
  Use this skill when managing a project's environment and dependencies with Pixi —
  authoring pixi.toml, mixing conda and PyPI dependencies, defining tasks, lock files,
  cross-platform environments, and CUDA toolkit setup. Reach for it any time you'd
  otherwise reach for conda, pip, or venv to set up or update a CV/ML environment, even
  if the user just says "add this dependency" or "set up the environment". For building
  and publishing the package to PyPI, see pypi.
---

# Pixi Skill

You are managing Python project environments and dependencies using pixi. Follow these patterns exactly.

## What Is Pixi and Why Use It

Pixi is a fast, cross-platform package manager built on the conda ecosystem. It replaces conda, mamba, pip, virtualenv, Make, and requirements.txt files with a single `pixi.toml` configuration file. It provides:

- **Reproducible environments**: Lock files ensure identical environments across machines.
- **Cross-platform support**: Works on Linux, macOS (Intel and Apple Silicon), and Windows.
- **Conda + PyPI integration**: Access packages from conda-forge, pytorch, and PyPI in the same environment.
- **Task runner**: Define project tasks (test, lint, train) directly in pixi.toml, replacing Makefiles.
- **Feature groups**: Organize optional dependencies into named features (dev, cuda, docs).
- **Fast resolution**: Uses the rattler solver for sub-second dependency resolution.

Every project in this framework uses pixi as the sole environment and dependency manager. Never use conda, pip, or virtualenv directly.

## pixi.toml Structure and Configuration

### Minimal pixi.toml

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

### Complete pixi.toml for a Training Project

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

## Dependencies Management

### Conda vs PyPI Dependencies

Pixi supports both conda-forge and PyPI packages. Use conda-forge for system-level packages and scientific libraries. Use PyPI for Python-only packages that are not on conda-forge or where the PyPI version is newer.

```toml
# Conda dependencies (from conda-forge or pytorch channels)
[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"              # System library, better from conda
ffmpeg = ">=6.0"              # System library, must be from conda
cuda-toolkit = ">=12.1"       # NVIDIA toolkit, only from conda

# PyPI dependencies (Python-only packages)
[pypi-dependencies]
torch = ">=2.2"               # PyTorch recommends pip install
lightning = ">=2.2"
albumentations = ">=1.3"
wandb = ">=0.16"
pydantic = ">=2.6"
timm = ">=0.9"
```

### Rules for Choosing Conda vs PyPI

1. **System libraries** (OpenCV, FFmpeg, CUDA): Always use conda.
2. **PyTorch ecosystem** (torch, torchvision, torchaudio): Use PyPI for reliable CUDA support.
3. **Pure Python packages** (pydantic, wandb, lightning): Either works; prefer PyPI for latest versions.
4. **Packages with C extensions** (Pillow, numpy, scipy): conda-forge often provides better-optimized builds.

### Version Pinning Strategy

```toml
# Pin major version for stability (allow minor/patch updates)
numpy = ">=1.26,<2"

# Pin minimum version only (for most packages)
pydantic = ">=2.6"

# Exact pin only for reproducibility-critical packages
# (use sparingly -- lock files handle this)
python = "3.11.*"
```

## Task Definitions

Tasks replace Makefiles and shell scripts. They are defined in pixi.toml and run with `pixi run <task>`.

### Simple Tasks

```toml
[tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
```

### Tasks with Dependencies

Tasks can depend on other tasks. Pixi runs dependencies first.

```toml
[tasks]
lint = "ruff check ."
format-check = "ruff format . --check"
typecheck = "mypy src/ --strict"
test = "pytest tests/ -v --cov=src --cov-fail-under=80"

# This runs lint, format-check, and typecheck before running tests
quality = { depends-on = ["lint", "format-check", "typecheck", "test"] }
```

### Tasks with Arguments and Environment Variables

```toml
[tasks]
train = { cmd = "python -m my_project.train", env = { WANDB_MODE = "online" } }
train-debug = { cmd = "python -m my_project.train", env = { WANDB_MODE = "disabled", CUDA_VISIBLE_DEVICES = "0" } }
serve = { cmd = "uvicorn my_project.api:app --host 0.0.0.0 --port 8000" }
```

### Feature-Specific Tasks

Tasks can be scoped to specific features, meaning they are only available when that feature is activated.

```toml
[feature.docs.tasks]
docs-build = "mkdocs build"
docs-serve = "mkdocs serve"

[feature.onnx.tasks]
export-onnx = "python scripts/export_onnx.py"
validate-onnx = "python scripts/validate_onnx.py"
```

## Feature Groups

Features let you organize optional dependencies into named groups. This keeps the default environment lean while allowing developers to opt into additional capabilities.

### Defining Features

```toml
# Core dependencies (always installed)
[dependencies]
python = ">=3.11"
numpy = ">=1.26"

# Development tools (installed in dev environments)
[feature.dev.dependencies]
pytest = ">=7.4"
ruff = ">=0.8"
mypy = ">=1.11"

# Documentation tools
[feature.docs.dependencies]
mkdocs = ">=1.5"
mkdocstrings = ">=0.24"

# CUDA-specific dependencies
[feature.cuda.dependencies]
cuda-toolkit = ">=12.1"

[feature.cuda.pypi-dependencies]
torch = { version = ">=2.2", extras = ["cuda"] }

# ONNX export dependencies
[feature.onnx.pypi-dependencies]
onnx = ">=1.15"
onnxruntime = ">=1.17"

# Experiment tracking
[feature.wandb.pypi-dependencies]
wandb = ">=0.16"
```

### Composing Environments from Features

```toml
[environments]
# Default: core + dev tools
default = { features = ["dev"], solve-group = "default" }

# Training: core + dev + cuda + wandb
train = { features = ["dev", "cuda", "wandb"], solve-group = "default" }

# CI: core + dev (no cuda, no wandb)
ci = { features = ["dev"], solve-group = "default" }

# Docs: core + dev + docs
docs = { features = ["dev", "docs"], solve-group = "default" }

# Export: core + dev + onnx
export = { features = ["dev", "onnx"], solve-group = "default" }
```

### Using Environments

```bash
# Install default environment
pixi install

# Install specific environment
pixi install -e train

# Run task in specific environment
pixi run -e train train
pixi run -e docs docs-serve
pixi run -e export export-onnx
```

## Environment Management

### Common Commands

```bash
# Install dependencies (creates/updates lock file)
pixi install

# Add a conda dependency
pixi add numpy ">=1.26"

# Add a PyPI dependency
pixi add --pypi torch ">=2.2"

# Add a dependency to a specific feature
pixi add --feature dev pytest ">=7.4"

# Remove a dependency
pixi remove numpy

# List all installed packages
pixi list

# Show environment info
pixi info

# Run a task
pixi run test
pixi run lint
pixi run quality

# Run a shell command in the environment
pixi shell

# Clean the environment (remove .pixi/)
pixi clean
```

### Lock File Management

Pixi generates `pixi.lock` which pins exact versions of all dependencies. This file should be committed to version control for applications but excluded for libraries.

```gitignore
# For applications (training projects, services): commit pixi.lock
# Do NOT add pixi.lock to .gitignore

# For libraries (published packages): ignore pixi.lock
pixi.lock
```

## Integration with pyproject.toml

Pixi can read project metadata from pyproject.toml, allowing the package definition and the environment definition to coexist.

### pyproject.toml with Pixi

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

## Project-Type Specific Configurations

### Training Project pixi.toml

```toml
[project]
name = "training-project"
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
timm = ">=0.9"
rich = ">=13.0"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[tasks]
train = "python -m training_project.train"
eval = "python -m training_project.evaluate"
test = "pytest tests/ -v --cov=src --cov-fail-under=80"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
```

### Inference Service pixi.toml

```toml
[project]
name = "inference-service"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"

[pypi-dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
fastapi = ">=0.109"
uvicorn = { version = ">=0.27", extras = ["standard"] }
pydantic = ">=2.6"
onnxruntime = ">=1.17"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
httpx = ">=0.27"

[tasks]
serve = "uvicorn inference_service.api:app --host 0.0.0.0 --port 8000"
serve-dev = "uvicorn inference_service.api:app --host 0.0.0.0 --port 8000 --reload"
test = "pytest tests/ -v"
lint = "ruff check ."
typecheck = "mypy src/ --strict"
```

### Library Package pixi.toml

```toml
[project]
name = "cv-library"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[feature.docs.dependencies]
mkdocs = ">=1.5"
mkdocstrings = ">=0.24"
mkdocs-material = ">=9.5"

[tasks]
test = "pytest tests/ -v --cov=src --cov-report=term --cov-fail-under=90"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
docs = "mkdocs serve"
docs-build = "mkdocs build"
```

## Anti-Patterns to Avoid

1. **Never use `pip install` directly** -- always use `pixi add --pypi`.
2. **Never use `conda install`** -- use `pixi add`.
3. **Never create requirements.txt** -- pixi.toml and pixi.lock replace it.
4. **Never create environment.yml** -- pixi.toml replaces it.
5. **Never use virtualenv or venv** -- pixi manages environments automatically.
6. **Never pin exact versions in pixi.toml** -- use minimum versions and let the lock file handle exact pinning.
7. **Never mix conda and pip channels for the same package** -- choose one source per package.
8. **Never forget to commit pixi.lock for applications** -- it ensures reproducibility.

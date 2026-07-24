# Configuration File Standards

Scope: the exact contents of the four configuration files every generated project
includes — `pyproject.toml`, `.pre-commit-config.yaml`, `pixi.toml`, and `.gitignore`.

## Contents

- [pyproject.toml (Always Include)](#pyprojecttoml-always-include)
- [Ruff Rule Categories Explained](#ruff-rule-categories-explained)
- [.pre-commit-config.yaml (Always Include)](#pre-commit-configyaml-always-include)
- [pixi.toml (Always Include)](#pixitoml-always-include)
- [.gitignore (Always Include)](#gitignore-always-include)

## pyproject.toml (Always Include)

This is the single source of truth for project metadata and tool configuration. Every
archetype includes this file with ruff, mypy, and pytest fully configured.

```toml
[project]
name = "{{project_slug}}"
version = "{{version}}"
description = "{{description}}"
authors = [{name = "{{author_name}}", email = "{{email}}"}]
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "S", "B", "A", "C4", "T20", "SIM"]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101"]

[tool.ruff.lint.isort]
known-first-party = ["{{package_name}}"]

[tool.mypy]
python_version = "3.11"
strict = true
disallow_any_explicit = true
warn_return_any = true
ignore_missing_imports = false

[[tool.mypy.overrides]]
module = ["cv2.*", "albumentations.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "--cov=src --cov-report=term --cov-report=xml --cov-fail-under=80"
```

## Ruff Rule Categories Explained

| Rule | Purpose |
|------|---------|
| `E` | pycodestyle errors |
| `F` | pyflakes (unused imports, variables) |
| `I` | isort (import ordering) |
| `N` | pep8-naming conventions |
| `UP` | pyupgrade (modern Python syntax) |
| `S` | bandit (security checks) |
| `B` | bugbear (common bugs) |
| `A` | builtins shadowing |
| `C4` | flake8-comprehensions |
| `T20` | flake8-print (no print statements) |
| `SIM` | flake8-simplify |

## .pre-commit-config.yaml (Always Include)

Pre-commit hooks run automatically before every commit to enforce code quality standards.
This configuration is non-negotiable for all archetypes.

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

See the **pre-commit** skill for the extended hook set (large-file blocking, secret
detection, YAML validation) and CI enforcement.

## pixi.toml (Always Include)

Pixi manages the project environment including conda and PyPI dependencies. This replaces
requirements.txt, environment.yml, and Makefiles.

```toml
[project]
name = "{{project_slug}}"
version = "{{version}}"
description = "{{description}}"
authors = ["{{author_name}} <{{email}}>"]
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
loguru = ">=0.7"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[tasks]
test = "pytest tests/ -v"
test-cov = "pytest tests/ --cov=src --cov-report=term --cov-report=html --cov-fail-under=80"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
```

## .gitignore (Always Include)

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Environments
.pixi/
.venv/
*.lock

# IDE
.vscode/
.idea/

# Testing
.coverage
htmlcov/
.pytest_cache/

# ML artifacts
*.pt
*.pth
*.onnx
*.safetensors
wandb/
mlruns/
lightning_logs/

# Data
data/raw/
data/processed/
*.hdf5
*.h5
```

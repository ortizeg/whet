# Pre-commit Hook Catalog

Scope: what each hook in a Python/ML pre-commit config does, how the Ruff and MyPy hooks
are configured, how to write custom local hooks, and how to run hooks manually.

## Contents

- [Why Pre-commit](#why-pre-commit)
- [Installation and Setup](#installation-and-setup)
- [Standard Hooks Explained](#standard-hooks-explained)
- [Ruff Integration](#ruff-integration)
- [MyPy Integration](#mypy-integration)
- [Custom Hooks](#custom-hooks)
- [File Exclusions](#file-exclusions)
- [Running Manually](#running-manually)

## Why Pre-commit

Pre-commit is a framework for managing and maintaining multi-language pre-commit hooks. It
runs a set of checks automatically before every `git commit`, catching issues early in the
development cycle before they reach code review or CI. For Python and machine learning
projects, pre-commit is essential for enforcing consistent code style, catching type
errors, preventing secrets from leaking, and keeping large files out of the repository.

Without pre-commit hooks, developers must remember to run formatters, linters, and type
checkers manually before committing. This leads to inconsistent code quality, noisy diffs
from formatting changes mixed with logic changes, and wasted CI minutes catching trivial
issues. Pre-commit solves all of these problems by automating checks at the point of
commit.

Key benefits:

- **Consistent code style** across all contributors without manual effort.
- **Early error detection** before code reaches CI pipelines.
- **Prevents accidental commits** of secrets, large files, or merge conflicts.
- **Faster code reviews** because reviewers focus on logic, not style.
- **Reproducible environments** with pinned hook versions.

## Installation and Setup

Install pre-commit using pip or pixi:

```bash
# Using pip
pip install pre-commit

# Using pixi (recommended for this project)
pixi add pre-commit --feature dev

# Verify installation
pre-commit --version
```

After installing, create a `.pre-commit-config.yaml` file at the root of your repository
and install the hooks:

```bash
# Install hooks into your .git directory
pre-commit install

# Also install commit-msg hooks if needed
pre-commit install --hook-type commit-msg
```

Once installed, pre-commit will run automatically on every `git commit`. Only the files
being committed (staged files) are checked, making it fast for incremental work.

## Standard Hooks Explained

The `pre-commit-hooks` repository provides general-purpose hooks:

- **trailing-whitespace**: Removes trailing whitespace from lines. Prevents noisy diffs.
- **end-of-file-fixer**: Ensures files end with a newline. Required by POSIX standard.
- **check-yaml**: Validates YAML syntax. Catches broken config files early.
- **check-added-large-files**: Prevents accidentally committing large files (models,
  datasets). The `--maxkb=5000` argument sets a 5 MB limit.
- **check-merge-conflict**: Detects leftover merge conflict markers (`<<<<<<<`).
- **detect-private-key**: Prevents committing private keys (SSH, RSA, etc.).

```yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.6.0
  hooks:
    - id: trailing-whitespace
    - id: end-of-file-fixer
    - id: check-yaml
    - id: check-added-large-files
      args: ['--maxkb=5000']
    - id: check-merge-conflict
    - id: detect-private-key
```

## Ruff Integration

Ruff is an extremely fast Python linter and formatter written in Rust. It replaces Flake8,
isort, Black, and many other tools in a single binary. The pre-commit configuration runs
two hooks:

### Ruff Linter

```yaml
- id: ruff
  args: [--fix, --exit-non-zero-on-fix]
```

The `--fix` flag automatically fixes safe issues (unused imports, import sorting). The
`--exit-non-zero-on-fix` flag ensures the hook fails when fixes are applied, so you can
review the changes before committing.

Configure Ruff in your `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # bugbear
    "SIM",  # simplify
    "ANN",  # annotations
    "S",    # bandit (security)
    "A",    # builtins
    "C4",   # comprehensions
    "DTZ",  # datetime
    "RUF",  # ruff-specific
]
ignore = ["ANN101", "ANN102"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "ANN"]
```

### Ruff Formatter

```yaml
- id: ruff-format
```

Ruff format is a drop-in replacement for Black. It formats code deterministically,
ensuring all contributors produce identical output. No additional configuration is needed
beyond `line-length` set above.

## MyPy Integration

MyPy performs static type checking on Python code. It catches type errors before runtime,
which is especially valuable in ML projects where tensor shape mismatches and incorrect
data types are common bugs.

```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.11.2
  hooks:
    - id: mypy
      additional_dependencies:
        - types-PyYAML
        - types-requests
        - pydantic
```

The `additional_dependencies` field installs type stubs that MyPy needs to check code that
uses those libraries. Without them, MyPy would report missing stub errors.

Configure MyPy in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = ["cv2.*", "albumentations.*", "torchvision.*"]
ignore_missing_imports = true
```

The `ignore_missing_imports` override is necessary for libraries that do not ship type
stubs, which is common in the computer vision ecosystem.

## Custom Hooks

You can define custom hooks for project-specific checks.

### No Large Files Hook (Custom)

```yaml
- repo: local
  hooks:
    - id: no-model-files
      name: Check for model files
      entry: bash -c 'for f in "$@"; do case "$f" in *.pt|*.pth|*.onnx|*.pkl|*.h5) echo "ERROR: Model file $f should not be committed. Use object storage or an artifact registry instead." && exit 1;; esac; done'
      language: system
      types: [file]
```

### Validate Hydra Configs

```yaml
- repo: local
  hooks:
    - id: validate-configs
      name: Validate Hydra configs
      entry: python -c "import yaml, sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]"
      language: python
      types: [yaml]
      additional_dependencies: [pyyaml]
```

## File Exclusions

Configure file exclusions for generated code, vendored files, or notebooks that should not
be checked:

```yaml
- id: ruff
  exclude: ^(notebooks/|generated/)
```

## Running Manually

You do not need to wait for a commit to run hooks. Manual execution is useful during
development and debugging:

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook on all files
pre-commit run ruff --all-files

# Run all hooks on specific files
pre-commit run --files src/model.py src/data.py

# Run a specific hook on specific files
pre-commit run mypy --files src/model.py

# Run hooks on staged files only (same as what runs on commit)
pre-commit run

# Update all hooks to their latest versions
pre-commit autoupdate

# Clean cached hook environments
pre-commit clean
```

---
name: code-quality
description: >
  Use this skill when setting up or fixing linting, type checking, and formatting for an
  AI/CV Python project — configuring Ruff rules, MyPy strict mode, resolving lint or type
  errors, and defining the quality bar. Reach for it any time you'd otherwise hand-tune
  pyproject linting config or silence a type error, even if the user just says "clean up
  the code" or "make it pass checks". This skill owns the standards themselves; the git
  commit-hook wiring lives in pre-commit and editor integration in vscode.
---

# Code Quality Skill

You are enforcing code quality standards for AI/CV Python projects. Follow these rules exactly.

## Core Philosophy

Code quality is not optional. Every file committed to the repository must pass linting, type checking, and formatting. This is enforced at three levels: the editor (pre-save), the commit (pre-commit hooks), and the CI pipeline (GitHub Actions). No exceptions.

## Ruff Configuration

Ruff is the single tool for both linting and formatting. It replaces flake8, isort, black, pyupgrade, bandit, and dozens of other tools.

### Standard ruff Configuration

```toml
# pyproject.toml

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src"]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "S",    # bandit (security)
    "B",    # bugbear
    "A",    # builtins shadowing
    "C4",   # flake8-comprehensions
    "T20",  # flake8-print
    "SIM",  # flake8-simplify
    "TCH",  # type-checking imports
    "RUF",  # ruff-specific rules
    "PTH",  # pathlib usage
    "ERA",  # eradicate (commented-out code)
]
ignore = [
    "E501",   # line length (handled by formatter)
    "S101",   # assert (allowed in tests via per-file-ignores)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["S101", "S105", "S106"]   # allow assert and hardcoded passwords in tests
"scripts/**/*.py" = ["T20"]                   # allow print in scripts
"notebooks/**/*.py" = ["T20", "E402"]         # allow print and late imports in notebooks

[tool.ruff.lint.isort]
known-first-party = ["{{package_name}}"]
force-single-line = false
lines-after-imports = 2

[tool.ruff.lint.pep8-naming]
classmethod-decorators = ["pydantic.field_validator", "pydantic.model_validator"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
line-ending = "auto"
docstring-code-format = true
```

### Ruff Rule Selection Rationale

| Rule | Why It Matters |
|------|---------------|
| `E` | Basic Python style errors. Catches obvious formatting issues. |
| `F` | Unused imports and variables. Dead code creates confusion. |
| `I` | Import ordering. Consistent imports improve readability. |
| `N` | Naming conventions. `snake_case` for functions, `PascalCase` for classes. |
| `UP` | Modern Python syntax. Use `dict` not `Dict`, `X \| Y` not `Union[X, Y]`. |
| `S` | Security issues. SQL injection, hardcoded passwords, unsafe YAML loading. |
| `B` | Common bugs. Mutable default arguments, unused loop variables. |
| `A` | Shadowing builtins. Never name a variable `input`, `list`, `type`, etc. |
| `C4` | Comprehension style. Use list comprehensions over `list(map(...))`. |
| `T20` | No print statements. Use `logging` instead. Print in prod code is a bug. |
| `SIM` | Simplification. Collapse nested ifs, use ternary where clearer. |
| `TCH` | Type-checking imports. Move type-only imports behind `TYPE_CHECKING`. |
| `RUF` | Ruff-specific. Catches additional patterns other tools miss. |
| `PTH` | Pathlib. Use `Path` not `os.path`. Modern Python file handling. |
| `ERA` | Dead code. Commented-out code should be deleted, not committed. |

### Running Ruff

```bash
ruff check .            # lint
ruff check . --fix      # lint + auto-fix
ruff format .           # format (add --check or --diff for CI)
```

## Mypy Configuration

Mypy provides static type checking. All projects use strict mode with no exceptions.

### Standard mypy Configuration

```toml
# pyproject.toml

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_any_generics = true
disallow_any_explicit = true
disallow_subclassing_any = true
disallow_untyped_calls = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
no_implicit_reexport = true
strict_equality = true
show_error_codes = true
show_column_numbers = true

[[tool.mypy.overrides]]
module = [
    "cv2.*",
    "albumentations.*",
    "torchvision.*",
    "timm.*",
    "wandb.*",
    "lightning.*",
    "tensorboard.*",
]
ignore_missing_imports = true
```

### Why Strict Mode?

Strict mode enables all of mypy's strictness flags at once. This catches:

- Functions without type annotations
- Variables with implicit `Any` types
- Missing return type annotations
- Untyped decorators
- Implicit `Optional` types

The override section exempts third-party libraries that do not ship type stubs. This list should be kept as small as possible -- if a library provides stubs, remove it from the override.

### Type Hint Rules

#### Always annotate function signatures

```python
# CORRECT: Fully annotated
def process_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    normalize: bool = True,
) -> np.ndarray:
    """Process a single image for model input."""
    ...

# WRONG: Missing annotations
def process_image(image, target_size, normalize=True):
    ...
```

#### Use modern type syntax (Python 3.11+)

```python
# CORRECT: built-in generics and PEP 604 unions
def get_labels() -> list[str]: ...
def get_config() -> dict[str, int]: ...
def maybe_transform(image: np.ndarray) -> np.ndarray | None: ...

# WRONG: legacy typing module — List[str], Dict[str, int], Optional[np.ndarray]
```

#### Use `from __future__ import annotations` in every file

```python
# ALWAYS include this as the first import
from __future__ import annotations

# This enables PEP 604 union syntax and deferred evaluation
# in all Python 3.11+ files
```

#### Type hint class attributes

```python
from __future__ import annotations

import torch.nn as nn


class Detector(nn.Module):
    """Object detection model."""

    backbone: nn.Module
    head: nn.Module
    num_classes: int

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self._build_backbone()
        self.head = self._build_head()
```

#### Use Protocol for structural typing

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageTransform(Protocol):
    """Protocol for image transformation functions."""

    def __call__(self, image: np.ndarray) -> np.ndarray: ...


def apply_transforms(
    image: np.ndarray,
    transforms: list[ImageTransform],
) -> np.ndarray:
    """Apply a sequence of transforms to an image."""
    for transform in transforms:
        image = transform(image)
    return image
```

#### Use TypeAlias for complex types

```python
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

# Define aliases for frequently used complex types
ImageArray: TypeAlias = np.ndarray
BoundingBox: TypeAlias = tuple[float, float, float, float]
BatchTensor: TypeAlias = torch.Tensor
DetectionList: TypeAlias = list[tuple[BoundingBox, float, int]]
```

### Running Mypy

```bash
mypy src/ --strict                          # type check
mypy src/ --strict --html-report report/    # optional HTML report
```

## Pre-commit Integration

Pre-commit hooks enforce standards automatically before every commit. No code reaches the repository without passing these checks.

### Standard .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ["--maxkb=5000"]
      - id: check-merge-conflict
      - id: debug-statements

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
        additional_dependencies:
          - types-PyYAML
          - types-requests
          - pydantic>=2.0
```

### Installing Pre-commit

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files (useful for first-time setup)
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

## CI Enforcement

GitHub Actions runs the same checks as pre-commit, ensuring nothing slips through even if a developer bypasses local hooks.

### Standard CI Workflow

```yaml
# .github/workflows/code-quality.yml
name: Code Quality

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.1

      - run: pixi run ruff check . --output-format=github
      - run: pixi run ruff format . --check
      - run: pixi run mypy src/ --strict
      - run: pixi run pytest tests/ -v --cov=src --cov-report=xml --cov-fail-under=80

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

## Common Code Patterns

### Logging Instead of Print

The `T20` rule bans `print()` in source. Use a logger with lazy `%`-style args
(the project convention is loguru — see the loguru skill):

```python
from loguru import logger

logger.info("Starting epoch {} with {} batches", epoch, num_batches)
logger.debug("Batch {}/{}, loss={:.4f}", batch_idx, num_batches, loss)
```

### Path Handling with pathlib

```python
from __future__ import annotations

from pathlib import Path


def find_images(data_dir: Path, extensions: tuple[str, ...] = (".jpg", ".png")) -> list[Path]:
    """Find all image files in a directory recursively."""
    images: list[Path] = []
    for ext in extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
    return sorted(images)


# CORRECT: pathlib
output_dir = Path("results") / "experiment_1"
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / "model.pt"

# WRONG: os.path
import os
output_dir = os.path.join("results", "experiment_1")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.pt")
```

### Error Messages

```python
# CORRECT: Assign message to variable, raise with variable
msg = f"Expected 3-channel image, got {image.shape[-1]} channels"
raise ValueError(msg)

# WRONG: Inline string in raise
raise ValueError(f"Expected 3-channel image, got {image.shape[-1]} channels")

# WRONG: No f-string when dynamic content is needed
raise ValueError("Invalid image shape")
```

## Quality Checklist

Before every commit, verify:

- [ ] `ruff check .` passes with zero violations
- [ ] `ruff format . --check` reports no changes needed
- [ ] `mypy src/ --strict` passes with zero errors
- [ ] `pytest tests/ -v --cov-fail-under=80` passes
- [ ] No `print()` statements in source code (use `logging`)
- [ ] No `os.path` usage (use `pathlib.Path`)
- [ ] No `typing.Dict`, `typing.List`, `typing.Optional` (use built-in generics)
- [ ] Every file starts with `from __future__ import annotations`
- [ ] Every function has complete type annotations
- [ ] Every class has a docstring
- [ ] Every public function has a docstring

## Anti-Patterns to Avoid

1. **Never disable rules globally** -- use per-file-ignores for specific exceptions.
2. **Never use `# type: ignore` without an error code** -- always specify `# type: ignore[specific-code]`.
3. **Never commit with pre-commit hooks disabled** -- fix the issues instead.
4. **Never use `Any` explicitly** -- find the correct type or use a Protocol.
5. **Never suppress linting warnings in CI** -- fix the code, not the tooling.
6. **Never use `noqa` without a specific code** -- always specify `# noqa: E501` not just `# noqa`.

# Pre-commit Troubleshooting

Scope: fixing the recurring failures — stale hook environments, MyPy import errors,
Ruff/MyPy conflicts, slow hooks, and safe ways to skip a hook.

## Contents

- [Hook Fails After Update](#hook-fails-after-update)
- [MyPy Reports Missing Imports](#mypy-reports-missing-imports)
- [Ruff and MyPy Conflict on Unused Imports](#ruff-and-mypy-conflict-on-unused-imports)
- [Hooks Are Slow](#hooks-are-slow)
- [Skipping Hooks Temporarily](#skipping-hooks-temporarily)

## Hook Fails After Update

When updating hook versions, cached environments may become stale:

```bash
# Clear all cached hook environments
pre-commit clean

# Reinstall hooks
pre-commit install
```

## MyPy Reports Missing Imports

If MyPy cannot find imports for your project's own modules, ensure the `mypy` hook knows
about your source layout:

```yaml
- id: mypy
  args: ["--namespace-packages", "--explicit-package-bases"]
  additional_dependencies:
    - types-PyYAML
    - types-requests
    - pydantic
```

Or set `MYPYPATH` in your environment.

Note that the mirrors-mypy hook runs in its own isolated virtualenv — it does not see your
project environment. Third-party libraries must be listed in `additional_dependencies` or
silenced with an `ignore_missing_imports` override in `pyproject.toml`.

## Ruff and MyPy Conflict on Unused Imports

Ruff may auto-remove imports that MyPy needs for type checking. Use `TYPE_CHECKING` blocks:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from torch import Tensor

def process(image_path: Path) -> Tensor:
    ...
```

## Hooks Are Slow

If hooks take too long, consider running only on changed files (the default) and moving
expensive checks like MyPy to CI only:

```yaml
- id: mypy
  stages: [manual]  # Only runs with: pre-commit run mypy --all-files
```

Other levers:

- Keep `check-added-large-files` and formatting hooks on every commit — they are
  milliseconds.
- Exclude generated directories and notebooks with `exclude:` rather than letting hooks
  walk them.
- `pre-commit clean` after a machine upgrade; a corrupted cached environment can make a
  fast hook rebuild itself on every run.

## Skipping Hooks Temporarily

During rapid prototyping, you may need to skip hooks:

```bash
# Skip all hooks for a single commit
git commit --no-verify -m "WIP: prototype"

# Skip specific hooks
SKIP=mypy git commit -m "Quick fix"
```

Use this sparingly. CI should still catch any issues.

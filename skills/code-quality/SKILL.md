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

You are enforcing code quality standards for AI/CV Python projects. Code quality is not
optional: every file committed must pass linting, type checking, and formatting, enforced
at the editor, the commit hook, and CI. This page holds the canonical config; the deep
dives below cover rule rationale, type-hint patterns, and violation fixes.

## Canonical Ruff configuration

Ruff is the single tool for both linting and formatting. It replaces flake8, isort, black,
pyupgrade, bandit, and dozens of other tools. Copy this block into `pyproject.toml`
verbatim — the `pre-commit` and `github-actions` skills defer to it.

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

## Canonical MyPy configuration

All projects use strict mode with no exceptions.

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

## Running the checks

```bash
ruff check .            # lint
ruff check . --fix      # lint + auto-fix
ruff format .           # format (add --check for CI)
mypy src/ --strict      # type check
```

## Quality checklist

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

## Anti-patterns to avoid

1. **Never disable rules globally** -- use per-file-ignores for specific exceptions.
2. **Never use `# type: ignore` without an error code** -- always specify `# type: ignore[specific-code]`.
3. **Never commit with pre-commit hooks disabled** -- fix the issues instead.
4. **Never use `Any` explicitly** -- find the correct type or use a Protocol.
5. **Never suppress linting warnings in CI** -- fix the code, not the tooling.
6. **Never use `noqa` without a specific code** -- always specify `# noqa: E501` not just `# noqa`.

## Deep dives

- `references/ruff-rules.md` — read when justifying, adding, or narrowing a rule family, or when deciding whether an exception belongs in `ignore` vs `per-file-ignores`.
- `references/mypy-strict.md` — read when writing annotations that satisfy strict mode: Protocols, TypeAlias, class attributes, modern generics, and the untyped-library override.
- `references/editor-and-ci-integration.md` — read when wiring these standards into commit hooks or a GitHub Actions quality job.
- `references/common-violations.md` — read when fixing a specific violation: `T20` print, `PTH` os.path, or inline raise messages.

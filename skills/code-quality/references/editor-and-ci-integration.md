# Enforcing the Standards at Commit and in CI

Scope: wiring the Ruff/MyPy standards into pre-commit hooks and a GitHub Actions job so
the same checks run locally and server-side.

## Contents

- [Standard .pre-commit-config.yaml](#standard-pre-commit-configyaml)
- [Installing Pre-commit](#installing-pre-commit)
- [Standard CI Workflow](#standard-ci-workflow)

Quality is enforced at three levels: the editor (pre-save), the commit (pre-commit hooks),
and the CI pipeline (GitHub Actions). The hook wiring in depth lives in the `pre-commit`
skill and editor setup in `vscode`; what follows is the minimum that carries these
standards across all three.

## Standard .pre-commit-config.yaml

Pre-commit hooks enforce standards automatically before every commit. No code reaches the
repository without passing these checks.

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

## Installing Pre-commit

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files (useful for first-time setup)
pre-commit run --all-files

# Run a specific hook
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

## Standard CI Workflow

GitHub Actions runs the same checks as pre-commit, ensuring nothing slips through even if a
developer bypasses local hooks.

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

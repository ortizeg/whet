# Pre-commit

The Pre-commit skill configures Git pre-commit hooks for automated code quality enforcement in CV/ML projects.

**Skill directory:** `skills/pre-commit/`

## Purpose

Pre-commit hooks catch problems before they enter version control: formatting issues, type errors, large binary files, and secrets. This skill teaches Claude Code to configure pre-commit with hooks relevant to ML projects, including ruff, mypy, Jupyter notebook cleaning, and large file prevention.

## When to Use

- Any project with more than one contributor
- Projects where code quality standards must be enforced consistently
- Repositories that might accidentally include large data files or model weights
- Projects using Jupyter notebooks alongside Python modules

## Key Patterns

### Pre-commit Configuration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=500']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [torch-stubs, types-PyYAML]
        args: [--strict]

  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
        description: Strip outputs from Jupyter notebooks
```

### Installation and Usage

```bash
# Install pre-commit (via Pixi)
pixi add pre-commit

# Install hooks
uv run pre-commit install

# Run on all files (first time or CI)
uv run pre-commit run --all-files
```

## Anti-Patterns to Avoid

- Do not skip hooks with `git commit --no-verify` to get around failures -- fix the issues
- Do not add hooks that take more than 30 seconds -- they discourage frequent commits
- Avoid running full test suites as pre-commit hooks -- run them in CI instead
- Do not forget to pin hook versions -- unpinned hooks break reproducibility

## Combines Well With

- **Code Quality** -- Ruff and mypy hooks enforce quality standards
- **GitHub Actions** -- Run `pre-commit run --all-files` in CI as a safety net
- **Testing** -- Quick smoke tests can be added as local hooks
- **Master Skill** -- Hooks enforce the conventions defined by the Master Skill

## Full Reference

See [`skills/pre-commit/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/pre-commit/SKILL.md) for patterns including custom local hooks for ML-specific validation, commit message linting, and large-artifact validation hooks.

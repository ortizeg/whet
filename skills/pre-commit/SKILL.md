---
name: pre-commit
description: >
  Use this skill when setting up git pre-commit hooks to enforce quality at commit time —
  authoring .pre-commit-config.yaml with Ruff, MyPy, YAML validation, large-file blocking,
  secret detection, and pre-commit.ci integration. Reach for it any time you want checks
  to run automatically on every commit, even if the user just says "add commit hooks" or
  "stop bad commits". This skill owns the hook wiring; the Ruff/MyPy standards themselves
  live in code-quality and the equivalent server-side checks in github-actions.
---

# Pre-commit Hooks for Python and ML Projects

Pre-commit runs a pinned set of checks before every `git commit`, on staged files only, so
style, type, secret, and large-file problems never reach review or CI. This page holds the
canonical config and the wiring you need almost every time; the deep dives cover the hook
catalog, CI enforcement, secret detection, and failure modes.

## Setup

```bash
pixi add pre-commit --feature dev   # or: pip install pre-commit
pre-commit install                  # hooks now run on every git commit
```

## The canonical .pre-commit-config.yaml

Suitable as-is for Python computer vision and machine learning projects. Every `rev` is
pinned for reproducibility.

```yaml
repos:
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

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - types-PyYAML
          - types-requests
          - pydantic
```

Three groups: general-purpose hygiene hooks, Ruff (lint + format, replacing Flake8, isort,
and Black), and MyPy for static type checking. `--exit-non-zero-on-fix` makes the commit
fail when Ruff rewrites a file so you review the change before committing.

## Running hooks

```bash
pre-commit run                       # staged files (what a commit runs)
pre-commit run --all-files           # whole repo
pre-commit run ruff --all-files      # one hook
pre-commit autoupdate                # bump pinned hook versions
```

## Conventions

1. **Pin all hook versions** using `rev` to ensure reproducibility.
2. **Run `pre-commit autoupdate` monthly** to pick up security fixes and new rules.
3. **Keep hooks fast**. Move slow checks (integration tests, full MyPy runs) to CI.
4. **Document your hooks** in the project README so new contributors know what to expect.
5. **Use `additional_dependencies`** for MyPy stubs rather than expecting them in the
   global environment.
6. **Always run pre-commit in CI** as a safety net against `--no-verify`.
7. **Configure file exclusions** for generated code, vendored files, or notebooks that
   should not be checked:

```yaml
- id: ruff
  exclude: ^(notebooks/|generated/)
```

## Anti-patterns

- **Unpinned or floating `rev`** — hook behaviour then changes under you; pin and update
  deliberately with `autoupdate`.
- **Local hooks only, no CI job** — `git commit --no-verify` silently bypasses everything;
  mirror the config in CI.
- **Slow hooks on every commit** — a full-repo MyPy run at commit time trains people to
  skip hooks; use `stages: [manual]` and enforce it in CI.
- **Relying on the ambient environment for MyPy stubs** — the mirrors-mypy hook runs in an
  isolated virtualenv; declare stubs in `additional_dependencies`.
- **Habitual `--no-verify`** — acceptable for a WIP prototype commit, never as a workflow.
- **No large-file or private-key hook** — a committed checkpoint or key cannot be removed
  from history without a rewrite; block it at commit time.
- **Duplicating tool settings in the hook `args`** — keep Ruff and MyPy configuration in
  `pyproject.toml` so the editor, CI, and the hook agree.

## Deep dives

- `references/hook-catalog.md` — read when choosing which hooks to include, configuring the Ruff or MyPy hooks and their `pyproject.toml` settings, writing a custom local hook, or looking up the full manual-run command set.
- `references/ci-integration.md` — read when adding the GitHub Actions pre-commit job, caching hook environments, or setting up pre-commit.ci.
- `references/secret-detection.md` — read when blocking credentials, private keys, or model artifacts from entering git history.
- `references/troubleshooting.md` — read when a hook fails after an update, MyPy can't resolve imports, Ruff strips imports MyPy needs, hooks are slow, or you need to skip one safely.

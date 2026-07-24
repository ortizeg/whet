# CLAUDE.md

This file provides context for Claude Code when working in this repository.

## Project Overview

**whet** — Sharpen your AI coder. A multi-platform CLI tool that installs expert
skills into AI coding agents (Claude Code, Google Antigravity, Cursor, Copilot).

Ships CV/ML as the flagship skill collection; the framework is domain-agnostic.

Distribution: `uvx whet` (one-shot) / `uv tool install whet` (permanent)

### What whet provides

- **32 Skills** — Best-practice knowledge modules (PyTorch Lightning, Pydantic, Docker, FastAPI, Hugging Face, AWS SageMaker, Gradio, Kubernetes, Model Evaluation, etc.)
- **6+ Archetypes** — Complete project templates for common CV/ML project types
- **CLI** — `whet add`, `whet install`, `whet list`, `whet search`, `whet doctor`
- **Multi-platform** — Claude Code, Google Antigravity, Cursor, GitHub Copilot

### Repository Structure

```
whet/
├── src/whet/              # Python package (src-layout, Typer CLI)
│   ├── cli/               # Command modules (install, skills, target, doctor, config)
│   ├── adapters/           # Platform adapters (claude, antigravity, cursor, copilot)
│   ├── core/              # Domain models (Pydantic: skill, config)
│   └── registry/          # Skill discovery, search, dependency resolution
├── skills/                # 25+ skill definitions (SKILL.md + README.md + skill.toml)
├── archetypes/            # 6+ project templates (README.md + archetype.toml + template/)
├── settings/              # Pre-built settings templates (claude.json, etc.)
├── docs/                  # MkDocs Material documentation
├── tests/                 # Self-tests (dynamic discovery, no hardcoded counts)
├── .github/workflows/     # CI/CD (test, lint, docs) — uses uv
├── pyproject.toml         # uv-managed, publishable to PyPI
├── justfile               # Task runner (wraps uv commands)
└── mkdocs.yml             # Documentation site config
```

## Development Commands

```bash
just test              # Run all tests
just lint              # Ruff linting
just typecheck         # MyPy strict type checking
just format-check      # Check formatting
just docs-build        # Build documentation site
just docs-serve        # Serve docs locally
just check             # Run all checks (lint + typecheck + test)
just dev               # Install in development mode
```

Or directly with uv:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy src/whet/ tests/ --strict
```

## How to Add a New Skill

1. Create `skills/<name>/SKILL.md` — YAML frontmatter (`name`, `description`), >500 chars, code examples and headers
2. Create `skills/<name>/skill.toml` — Machine-readable metadata (category, tier, tags, deps, compatibility)
3. Create `skills/<name>/README.md` — Must explain purpose (include words like "when", "use", "purpose")
4. Create `docs/skills/<name>.md` — Documentation page (Purpose, When to Use, Key Patterns, Anti-Patterns)
5. Add nav entry to `mkdocs.yml` under Skills section

Tests discover skills dynamically — no hardcoded lists to update.

### Writing the `description` (this is the trigger)

Only the frontmatter is preloaded; the `description` is the entire discovery mechanism.
Write it third-person and trigger-first, and be a little "pushy":

```
description: >
  Use this skill when <concrete situations>. Reach for it any time you would otherwise
  <the manual thing>, even if the user doesn't say "<tool>" explicitly. Not for
  <adjacent thing> (see <other-skill>).
```

Always disambiguate against adjacent skills so two skills never compete for the same trigger.

### Progressive disclosure (keep SKILL.md thin)

A SKILL.md loads fully into context when triggered and **stays there for the session**, so
every line is a recurring token cost. Structure a skill as an index plus on-demand detail:

```
skills/<name>/
├── SKILL.md          # ~120-200 line index: core patterns, conventions, anti-patterns
└── references/       # topic files, loaded only when needed
    ├── <topic-a>.md
    └── <topic-b>.md
```

Rules:
- Keep `SKILL.md` **under 500 lines** (target 120–200 for a split skill).
- Keep the 80%-case patterns, conventions, and anti-patterns **inline**; move depth to `references/`.
- End `SKILL.md` with a `## Deep dives` list giving each reference a *"read this when…"* trigger.
- **One level deep only** — a reference file must never link to another reference file.
- Reference files over 100 lines start with a table of contents; name them descriptively.

The installer copies `references/` for directory-based platforms (Claude, Antigravity) and
inlines them for flat-file platforms (Cursor, Copilot), so no content is lost either way.

### Core vs extra tier

`tier = "core"` (default) installs with `whet install`. `tier = "extra"` marks a skill as
opt-in — it is skipped unless the user passes `--include-extras`. Use `extra` for skills
that are real but outside the flagship path, so they don't dilute the default trigger surface.

## Companion skills (not shipped by whet)

whet does not ship a general product-UI ruleset — `gradio` covers ML demos only. For
application/dashboard UI work, use the external **`interface-design`** skill alongside whet.
If its `references/` directory is missing from your install, its "Deep dives" links will not
resolve — reinstall it with the reference files present.

## How to Add a New Archetype

1. Create `archetypes/<name>/README.md` — Must be >500 chars, include code blocks
2. Create `archetypes/<name>/archetype.toml` — Archetype metadata (category, skills)
3. Create `archetypes/<name>/template/` directory with template files
4. Create `docs/archetypes/<name>.md` — Documentation page
5. Add nav entry to `mkdocs.yml` under Archetypes section

## Skill File Format

### SKILL.md (Universal Standard — read by LLMs)

```yaml
---
name: skill-name
description: >
  Concise description for LLM discovery.
---

# Skill Title
(skill content...)
```

### skill.toml (Machine-readable metadata — NOT sent to LLMs)

```toml
[skill]
name = "skill-name"
version = "1.0.0"
category = "cv-ml"  # core | cv-ml | infra | cloud | experiment-tracking
tier = "core"       # core (default, installed) | extra (opt-in via --include-extras)
tags = ["tag1", "tag2"]

[dependencies]
requires = ["other-skill"]
recommends = ["another-skill"]

[compatibility]
python = ">=3.11"
libraries = { some-lib = ">=1.0" }
```

## Conventions

- **Python 3.11+** — minimum version, use modern syntax (PEP 604 unions, etc.)
- **Ruff** — linting and formatting (line-length 100, rules: E, F, I, N, UP, S, B, A, C4, T20, SIM)
- **MyPy strict** — all code must pass `mypy --strict`
- **uv** — dependency and environment management
- **Loguru** — mandatory logging convention across all skills
- **Pydantic V2** — `BaseModel` for all configs and data structures
- **src-layout** — mandatory for all project archetypes and the whet package itself

## CI/CD

Three GitHub Actions workflows, all must pass before merge:

1. **test.yml** — `uv run pytest`, `uv run ruff check`, `uv run mypy --strict`
2. **lint.yml** — `ruff format --check`, linting, type checking, YAML validation
3. **docs.yml** — Builds and deploys MkDocs to GitHub Pages (main branch only)

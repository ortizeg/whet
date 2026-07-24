# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the
version is below 1.0.0, breaking changes are released in a minor bump.

## [0.3.0] - 2026-07-24

A library-wide audit of all three pillars — skills, agents, and archetypes — measured
against Anthropic's Agent Skills guidance. Always-resident skill context is down 73%,
the agents concept is gone, every archetype generates a project that runs, and the
authoring rules are now enforced in CI rather than by discipline.

### Removed

- **The `agents/` directory and the agent concept.** The six "agents" were never Claude
  Code subagents: four had no YAML frontmatter (so they installed with their H1 as the
  entire description and were effectively untriggerable), all six installed into
  `.claude/skills/` rather than `.claude/agents/`, `agent.toml` was never parsed by any
  code path, and both "blocking" `action.yml` files ran `pixi` against a repo with no
  `pixi.toml`. Roughly 77% of their content duplicated existing skills, often diverging
  from them. GSD owns agency; whet owns domain knowledge, and knowledge is a skill.
- **`whet install --agents-only` and `--skills-only` flags**, along with
  `discover_agents()` and the `agents_dir` config field.
- **Skills `dvc` and `cv-model-selection`.** Guidance that referenced DVC as a generic
  concept was rewritten tool-neutral rather than left pointing at a deleted skill.

### Added

- **`model-evaluation`** — detection mAP/IoU via `supervision`, confusion matrices,
  per-class and per-size breakdowns, deployment-threshold selection, per-slice failure
  analysis, and eval-as-CI regression gates.
- **`pydantic-ai`** — typed LLM/VLM structured outputs and VLM-in-the-loop auto-labeling.
- **`data-pipelines`** — storage-format selection, group-aware leakage-preventing dataset
  splitting, schema evolution, and data-quality validation.
- **`whet install --prune`** — removes skills that no longer exist upstream. Scoped by a
  `.whet-manifest.json` recording what whet installed, so skills placed in the same
  directory by other tools are never touched.
- **`whet install --include-extras`** and a `tier` field in `skill.toml`. Skills outside
  the flagship path are opt-in: `aws-sagemaker`, `github-repo-setup`, `gradio`,
  `huggingface`, `kubernetes`, `mlflow`, `vscode`.
- **`whet init --with-recommended` / `--no-skills`.**
- **Progressive disclosure** across the library: 29 of 32 skills are now a thin index
  plus a `references/` directory loaded on demand — 155 reference files in total.
- **CI enforcement.** A new `archetypes` job renders every archetype and runs the
  toolchain against the generated project, plus guards for skill-authoring rules
  (line ceiling, reference-link integrity, orphaned references, one-level-deep
  references, tables of contents, trigger-first and third-person descriptions,
  listing budget) and for archetype composition and documented directory trees.

### Changed

- **`whet init` now installs the archetype's skills** (with their `references/`) instead
  of printing a `whet add ...` hint. An archetype's value over a folder copy is the skill
  set it composes, so leaving that as homework made `[skills]` advisory.
- **Skill `pydantic-strict` renamed to `pydantic`.**
- **All skill descriptions rewritten** as third-person, trigger-first "use this skill
  when…" statements with explicit disambiguation between adjacent skills. They were topic
  summaries, which under-trigger.
- **Every archetype template now generates a project that runs.** Previously four of six
  generated only a four-file stub, and the two with templates produced projects that could
  not import their own entry point, crashed on the first training step, or shipped no ONNX
  despite being named for it. Template files went from 19 to 125.
- **pixi is the canonical environment manager** for generated projects; manifests migrated
  from the deprecated `[project]` table to `[workspace]`.
- **Archetype skill composition** now reflects what each template actually ships — every
  archetype requires `pixi` and `code-quality`, and `cv-inference-service` finally
  requires `fastapi`.

### Fixed

- Template files under a `data/` path were silently excluded by the repository's own
  `.gitignore`, so a clean checkout produced projects missing `data/raw`, `data/processed`,
  and an entire Hydra `data` config group.
- `.ipynb` files were not treated as text by the scaffold engine, so `${package_name}`
  inside a notebook was never substituted.
- Adapters now install a skill's `references/` directory (Claude, Antigravity) or inline
  it (Cursor, Copilot), so deep-dive links resolve on every platform.
- Documented directory trees in archetype READMEs and docs pages are regenerated from the
  templates; 78 documented-but-absent files were removed.

## [0.2.1] - 2026-07-23

Published to PyPI outside this repository's tag-based release workflow; no corresponding
git tag exists. Recorded here for continuity.

## [0.1.0] - 2026-02-09

Initial whet CLI implementation.

[0.3.0]: https://github.com/ortizeg/whet/releases/tag/v0.3.0

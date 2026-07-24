---
name: github-actions
description: >
  Use this skill when creating or fixing CI/CD workflows for an ML/CV repo on GitHub
  Actions — tiered lint/test/build/deploy pipelines, GPU runners, dependency caching,
  matrix builds, Docker build-and-push, releases, and AI-agent integration. Reach for it
  any time you'd otherwise hand-write a .github/workflows YAML or debug a failing
  pipeline, even if the user just says "set up CI" or "why is the build red". For hooks
  that run locally before commit see pre-commit; for repo settings and branch protection
  see github-repo-setup.
---

# GitHub Actions Skill

CI/CD workflow patterns for ML/CV projects: tiered pipelines, GPU runners, caching, and
AI agent integration. This page holds the setup block, the first workflow tier, and the
green-CI discipline you need almost every time; the deep dives below cover matrices,
GPU runners, Docker, and releases.

**Scope boundary:** this skill owns *server-side* CI. Hooks that run locally before a
commit belong to the `pre-commit` skill; the Ruff/MyPy rule sets and quality bar
themselves belong to the `code-quality` skill. This skill only *runs* those checks.

## Standard pixi Setup

Every job shares the same setup steps. Later examples — here and in the deep dives —
abbreviate this as `# ...pixi setup...`.

```yaml
- uses: actions/checkout@v4
- uses: prefix-dev/setup-pixi@v0.8.1
  with:
    cache: true   # caches on pixi.lock hash
- run: pixi install
```

## Workflow Tiers

Decide what runs at each trigger before writing YAML:

```
What should CI do?
├── Every commit → lint + type check + unit tests        (< 5 min)
├── Every PR     → above + integration tests + Docker build (< 15 min)
├── Merge to main → above + push image + deploy staging
└── Release tag  → deploy production + create GitHub Release
```

Organize workflows by speed and trigger frequency. Tier 1 is the one every repo needs:

```yaml
# .github/workflows/lint.yml
name: Lint & Format
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      # ...pixi setup...
      - run: pixi run ruff format --check .
      - run: pixi run ruff check .
      - run: pixi run mypy src/
```

## CI Verification (Required Before Task Completion)

A PR is **not done** until CI is green. Always verify after pushing:

```bash
gh pr checks <PR_NUMBER>          # check status
gh run view <RUN_ID> --log-failed # read failing logs
```

Workflow: run local checks (`ruff format && ruff check && pytest`), push, run
`gh pr checks <PR#>`, fix any failure locally, push, repeat until all checks are green.

### Common CI Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `ruff format --check` fails | Pre-commit ruff version differs from project ruff | Sync `.pre-commit-config.yaml` rev to `pixi run ruff --version` |
| `ruff check` fails | New lint violations | Run `pixi run ruff check .` locally and fix |
| `mypy` fails | Type errors | Run `pixi run mypy src/` locally and fix |
| `pytest` fails | Test failures | Run `pixi run pytest` locally and fix |
| Merge conflicts | Branch diverged from base | Merge/rebase base, resolve, re-run |

Keep formatter/linter versions synchronized: the `.pre-commit-config.yaml` ruff rev must
match `pixi run ruff --version`. If pre-commit reformats files that CI then rejects, the
versions are out of sync.

## Conventions

1. **Use pixi in CI** — keep CI commands identical to local development.
2. **Cache aggressively** — pixi environments, model weights.
3. **Fail fast on lint** — run lint before tests for quick feedback.
4. **Pin action versions** — use `@v4`, not `@main`.
5. **Set timeouts** — prevent runaway training jobs.
6. **Use artifacts wisely** — set retention days; don't upload huge datasets.
7. **Protect secrets** — GitHub Secrets only, never hardcode.
8. **Gate production** — require approval via environments.
9. **Verify CI after every push** — never call a task done until checks are green; fix broken CI immediately.

## Anti-patterns

- **Floating action tags** — `uses: actions/checkout@main` breaks without warning; pin to `@v4`.
- **Hardcoded credentials** — API keys belong in GitHub Secrets and are passed via `env`, never inlined in YAML.
- **No timeout on training jobs** — a hung GPU job occupies a self-hosted runner indefinitely; always set `timeout-minutes`.
- **Tests before lint** — running slow tests first delays feedback on a one-line formatting failure.
- **Uploading datasets as artifacts** — artifacts are for checkpoints and reports; set `retention-days` and keep them small.
- **Unprotected production deploys** — a deploy job with no `environment:` gate can ship on any merge.
- **Drifting tool versions** — a `.pre-commit-config.yaml` ruff rev that differs from the pixi ruff makes local and CI formatting disagree forever.
- **Calling a task done on a red build** — an unverified push is an unfinished task.

## Deep dives

- `references/workflow-tiers.md` — read when building out the full pipeline beyond lint: PR test workflows, GPU training validation, reusable `workflow_call` workflows, secrets/environments, or PR quality-gate agent jobs.
- `references/matrix-and-caching.md` — read when a job needs an OS/Python or CPU/GPU matrix, model-weight or pip caching, or must hand artifacts to a downstream job.
- `references/gpu-runners.md` — read when a workflow needs a self-hosted GPU runner or CUDA device setup.
- `references/docker-build-and-push.md` — read when CI must build a Docker image and push it to GHCR with layer caching.
- `references/releases.md` — read when automating version bumps, changelogs, PyPI publishing, or GitHub Releases with semantic release.

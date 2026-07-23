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

CI/CD workflow patterns for ML/CV projects: tiered pipelines, GPU runners, caching, and AI agent integration.

## Standard pixi Setup

Every job shares the same setup steps. Later examples abbreviate this as `# ...pixi setup...`.

```yaml
- uses: actions/checkout@v4
- uses: prefix-dev/setup-pixi@v0.8.1
  with:
    cache: true   # caches on pixi.lock hash
- run: pixi install
```

## Workflow Tiers

Organize workflows by speed and trigger frequency.

### Tier 1: Lint and Format (every push)

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

### Tier 2: Tests (pull requests)

Adds a Python matrix and coverage upload.

```yaml
# .github/workflows/test.yml
name: Test Suite
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      # ...pixi setup...
      - run: pixi run pytest --cov=src --cov-report=xml --cov-fail-under=80 -v
      - name: Upload coverage
        if: always()
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false
```

### Tier 3: Training Validation (merge to main, GPU runner)

Runs on a self-hosted GPU runner with a timeout and artifact upload.

```yaml
# .github/workflows/train-validation.yml
name: Training Validation
on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      epochs:
        default: "2"
jobs:
  smoke-test:
    runs-on: [self-hosted, gpu, linux]
    timeout-minutes: 60
    steps:
      # ...pixi setup...
      - name: Run smoke training
        run: |
          pixi run python -m my_project.train \
            trainer.max_epochs=${{ github.event.inputs.epochs || '2' }} \
            data.batch_size=4
      - name: Upload training artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: training-artifacts
          path: |
            outputs/
            checkpoints/
          retention-days: 7
```

## Matrix Testing

OS + Python matrix (use `include`/`exclude` to prune combinations):

```yaml
jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.11", "3.12"]
        exclude:
          - os: macos-latest
            python-version: "3.12"
    runs-on: ${{ matrix.os }}
```

CPU/GPU matrix — pair a runner label with a device flag via `include`:

```yaml
jobs:
  test:
    strategy:
      matrix:
        include:
          - runner: ubuntu-latest
            device: cpu
          - runner: self-hosted-gpu
            device: cuda
    runs-on: ${{ matrix.runner }}
    steps:
      - run: pixi run pytest -v --device=${{ matrix.device }}
```

## Caching

`setup-pixi` with `cache: true` handles the environment. Add caches for pip fallback and model weights keyed on the relevant file hash:

```yaml
- name: Cache pretrained models
  uses: actions/cache@v4
  with:
    path: ~/.cache/torch/hub
    key: ${{ runner.os }}-torch-hub-${{ hashFiles('configs/model.yaml') }}
    restore-keys: |
      ${{ runner.os }}-torch-hub-
```

The same pattern works for `~/.cache/pip` keyed on `requirements*.txt`.

## GPU Runners

Label self-hosted runners by capability (`self-hosted`, `gpu`, `linux`, or a GPU type like `a100`), then target them and verify the device:

```yaml
jobs:
  train:
    runs-on: [self-hosted, gpu, linux]
    steps:
      - name: Verify GPU
        run: nvidia-smi
      - name: Set CUDA device
        run: echo "CUDA_VISIBLE_DEVICES=0" >> $GITHUB_ENV
```

## Artifacts Across Jobs

Upload in one job (set `retention-days`), download in a dependent job via `needs`:

```yaml
jobs:
  train:
    steps:
      - uses: actions/upload-artifact@v4
        with:
          name: model-${{ github.sha }}
          path: checkpoints/
          retention-days: 30
  evaluate:
    needs: train
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: model-${{ github.sha }}
          path: checkpoints/
```

## Reusable Workflows

Define with `workflow_call`, call with `uses:` and pass `inputs`:

```yaml
# .github/workflows/reusable-lint.yml
on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: "3.11"
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      # ...pixi setup...
      - run: pixi run ruff check .
      - run: pixi run mypy src/
```

```yaml
# .github/workflows/ci.yml — caller
jobs:
  lint:
    uses: ./.github/workflows/reusable-lint.yml
    with:
      python-version: "3.11"
  test:
    needs: lint
    uses: ./.github/workflows/reusable-test.yml
```

## Secrets and Environments

Pass repository secrets via `env`; gate production jobs behind a protected `environment` that requires approval:

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production   # requires manual approval
    steps:
      - name: Train + deploy
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: pixi run deploy
```

## Agent Integration

Run quality-gate agents on PRs. Code review and test-engineer agents share the same shape — the review agent runs format/lint/mypy plus security (`ruff check --select S`); the test agent runs coverage and flags skipped tests:

```yaml
# .github/workflows/code-review.yml
name: Code Review
on:
  pull_request:
    types: [opened, synchronize]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      # ...pixi setup...
      - run: pixi run ruff format --check .
      - run: pixi run ruff check .
      - run: pixi run mypy src/
      - name: Security checks
        run: pixi run ruff check --select S .
      - name: Warn on skipped tests
        run: |
          if pixi run pytest --collect-only -q 2>&1 | grep -q "skipped"; then
            echo "::warning::Skipped tests found. Fix or remove them."
          fi
```

## Docker Build and Push

Build a multi-stage image (`target: inference`) and push to GHCR with GHA layer caching:

```yaml
# .github/workflows/docker.yml
name: Build Docker Image
on:
  push:
    branches: [main]
    tags: ["v*"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          target: inference
```

## Semantic Release

Automate version bumps, changelogs, and publishing. The release job runs after checks pass, analyzes commits since the last tag, and publishes only when a new version is cut.

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    branches: [main, develop]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      # ...pixi setup...
      - run: pixi run ruff check .
      - run: pixi run mypy src/
      - run: pixi run pytest
  release:
    runs-on: ubuntu-latest
    concurrency: release
    environment: pypi
    needs: [check]
    if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
    permissions:
      id-token: write    # OIDC / trusted publishing
      contents: write    # create releases/tags
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # full history for commit analysis
      - name: Python Semantic Release
        id: release
        uses: python-semantic-release/python-semantic-release@v9.15.1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
      - name: Publish to PyPI
        if: steps.release.outputs.released == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
      - name: Publish to GitHub Releases
        if: steps.release.outputs.released == 'true'
        uses: python-semantic-release/upload-to-gh-release@v9.15.1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          tag: ${{ steps.release.outputs.tag }}
```

Key details:

- `fetch-depth: 0` — semantic release needs full git history to analyze commits since the last tag.
- `concurrency: release` — prevents parallel release jobs from conflicting.
- `environment: pypi` — attach deployment protection rules.
- Conditional publish (`released == 'true'`) — only publishes when a new version is actually created.
- Prefer trusted publishing (`id-token: write` + OIDC) over long-lived API tokens.

Branch strategy — `main` cuts production releases (`1.2.0`); `develop` cuts prereleases (`1.3.0-dev.1`). Configure in `pyproject.toml` (see PyPI skill):

```toml
[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

## CI Verification (Required Before Task Completion)

A PR is **not done** until CI is green. Always verify after pushing:

```bash
gh pr checks <PR_NUMBER>          # check status
gh run view <RUN_ID> --log-failed # read failing logs
```

Workflow: run local checks (`ruff format && ruff check && pytest`), push, run `gh pr checks <PR#>`, fix any failure locally, push, repeat until all checks are green.

### Common CI Failures

| Failure | Cause | Fix |
|---------|-------|-----|
| `ruff format --check` fails | Pre-commit ruff version differs from project ruff | Sync `.pre-commit-config.yaml` rev to `pixi run ruff --version` |
| `ruff check` fails | New lint violations | Run `pixi run ruff check .` locally and fix |
| `mypy` fails | Type errors | Run `pixi run mypy src/` locally and fix |
| `pytest` fails | Test failures | Run `pixi run pytest` locally and fix |
| Merge conflicts | Branch diverged from base | Merge/rebase base, resolve, re-run |

Keep formatter/linter versions synchronized: the `.pre-commit-config.yaml` ruff rev must match `pixi run ruff --version`. If pre-commit reformats files that CI then rejects, the versions are out of sync.

## Best Practices

1. **Use pixi in CI** — keep CI commands identical to local development.
2. **Cache aggressively** — pixi environments, model weights.
3. **Fail fast on lint** — run lint before tests for quick feedback.
4. **Pin action versions** — use `@v4`, not `@main`.
5. **Set timeouts** — prevent runaway training jobs.
6. **Use artifacts wisely** — set retention days; don't upload huge datasets.
7. **Protect secrets** — GitHub Secrets only, never hardcode.
8. **Gate production** — require approval via environments.
9. **Verify CI after every push** — never call a task done until checks are green; fix broken CI immediately.

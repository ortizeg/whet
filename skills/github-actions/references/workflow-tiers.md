# Workflow Tiers

Scope: the full tiered CI pipeline for an ML/CV repo — lint on every push, tests on pull
requests, GPU training validation on merge to main — plus reusable workflows, secrets and
environments, and quality-gate agent workflows.

All examples abbreviate the shared setup as `# ...pixi setup...`, which expands to:

```yaml
- uses: actions/checkout@v4
- uses: prefix-dev/setup-pixi@v0.8.1
  with:
    cache: true   # caches on pixi.lock hash
- run: pixi install
```

## Contents

- [Choosing Tiers](#choosing-tiers)
- [Tier 1: Lint and Format (every push)](#tier-1-lint-and-format-every-push)
- [Tier 2: Tests (pull requests)](#tier-2-tests-pull-requests)
- [Tier 3: Training Validation (merge to main, GPU runner)](#tier-3-training-validation-merge-to-main-gpu-runner)
- [Reusable Workflows](#reusable-workflows)
- [Secrets and Environments](#secrets-and-environments)
- [Agent Integration](#agent-integration)

## Choosing Tiers

Decide what runs at each trigger before writing YAML:

```
What should CI do?
├── Every commit → lint + type check + unit tests        (< 5 min)
├── Every PR     → above + integration tests + Docker build (< 15 min)
├── Merge to main → above + push image + deploy staging
└── Release tag  → deploy production + create GitHub Release
```

Organize workflows by speed and trigger frequency.

## Tier 1: Lint and Format (every push)

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

## Tier 2: Tests (pull requests)

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

## Tier 3: Training Validation (merge to main, GPU runner)

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

Pass repository secrets via `env`; gate production jobs behind a protected `environment`
that requires approval:

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

Run quality-gate agents on PRs. Code review and test-engineer agents share the same
shape — the review agent runs format/lint/mypy plus security (`ruff check --select S`);
the test agent runs coverage and flags skipped tests:

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

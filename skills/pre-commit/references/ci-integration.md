# Running Pre-commit in CI

Scope: enforcing the same hooks server-side so `--no-verify` cannot bypass them, with a
cached GitHub Actions workflow.

## Why CI Enforcement

Pre-commit should also run in CI to catch cases where developers bypass local hooks (using
`--no-verify`), or where a contributor never ran `pre-commit install` at all. CI is the
safety net; the local hook is the fast feedback loop.

## GitHub Actions Workflow

```yaml
# .github/workflows/pre-commit.yml
name: Pre-commit

on:
  pull_request:
  push:
    branches: [main]

jobs:
  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install pre-commit
        run: pip install pre-commit

      - name: Cache pre-commit hooks
        uses: actions/cache@v4
        with:
          path: ~/.cache/pre-commit
          key: pre-commit-${{ hashFiles('.pre-commit-config.yaml') }}

      - name: Run pre-commit
        run: pre-commit run --all-files --show-diff-on-failure
```

The `--show-diff-on-failure` flag displays exactly what the formatter would change, making
it easy to fix issues locally.

## Notes

- **Cache key on the config file** — `hashFiles('.pre-commit-config.yaml')` invalidates the
  hook-environment cache exactly when hook versions change, which is the only time the
  environments need rebuilding.
- **`--all-files` in CI, staged files locally** — CI checks the whole repository so a hook
  added later is enforced on existing code, not only on newly touched files.
- **Manual-stage hooks still need a job** — hooks marked `stages: [manual]` (for example a
  slow full MyPy run) do not run under `pre-commit run --all-files`; invoke them
  explicitly with `pre-commit run mypy --hook-stage manual --all-files` or run the tool
  directly in its own CI step.
- **pre-commit.ci** is the hosted alternative: it runs the same config on every pull
  request and opens autoupdate PRs on a schedule, replacing the workflow above.

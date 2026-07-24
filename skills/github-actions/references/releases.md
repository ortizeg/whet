# Semantic Release and Publishing

Scope: automating version bumps, changelogs, PyPI publishing, and GitHub Releases with
python-semantic-release, including the main/develop branch strategy.

## Contents

- [Release Workflow](#release-workflow)
- [Key Details](#key-details)
- [Branch Strategy](#branch-strategy)

## Release Workflow

Automate version bumps, changelogs, and publishing. The release job runs after checks
pass, analyzes commits since the last tag, and publishes only when a new version is cut.

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

The `# ...pixi setup...` placeholder above expands to:

```yaml
- uses: actions/checkout@v4
- uses: prefix-dev/setup-pixi@v0.8.1
  with:
    cache: true   # caches on pixi.lock hash
- run: pixi install
```

## Key Details

- `fetch-depth: 0` — semantic release needs full git history to analyze commits since the last tag.
- `concurrency: release` — prevents parallel release jobs from conflicting.
- `environment: pypi` — attach deployment protection rules.
- Conditional publish (`released == 'true'`) — only publishes when a new version is actually created.
- Prefer trusted publishing (`id-token: write` + OIDC) over long-lived API tokens.

## Branch Strategy

`main` cuts production releases (`1.2.0`); `develop` cuts prereleases (`1.3.0-dev.1`).
Configure in `pyproject.toml` (see PyPI skill):

```toml
[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.branches.develop]
match = "develop"
prerelease = true
prerelease_token = "dev"
```

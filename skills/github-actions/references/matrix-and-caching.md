# Matrix Builds, Caching, and Artifacts

Scope: OS/Python and CPU/GPU matrix strategies, dependency and model-weight caching, and
passing artifacts between jobs.

## Contents

- [Matrix Testing](#matrix-testing)
- [Caching](#caching)
- [Artifacts Across Jobs](#artifacts-across-jobs)

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

`setup-pixi` with `cache: true` handles the environment. Add caches for pip fallback and
model weights keyed on the relevant file hash:

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

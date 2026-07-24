---
name: pixi
description: >
  Use this skill when managing a project's environment and dependencies with Pixi —
  authoring pixi.toml, mixing conda and PyPI dependencies, defining tasks, lock files,
  cross-platform environments, and CUDA toolkit setup. Reach for it any time you'd
  otherwise reach for conda, pip, or venv to set up or update a CV/ML environment, even
  if the user just says "add this dependency" or "set up the environment". For building
  and publishing the package to PyPI, see pypi.
---

# Pixi Skill

Pixi is a fast, cross-platform package manager built on the conda ecosystem. It replaces
conda, mamba, pip, virtualenv, Make, and requirements.txt with a single `pixi.toml`:
reproducible lock files, conda + PyPI in one environment, a built-in task runner, and
named feature groups. Every project in this framework uses pixi as the sole environment
and dependency manager — never conda, pip, or virtualenv directly.

## The manifest core

A working `pixi.toml` needs a `[workspace]` block naming the channels and platforms, a
`[dependencies]` block for conda packages, and (usually) `[pypi-dependencies]` and
`[tasks]`.

```toml
[workspace]
name = "my-cv-project"
version = "0.1.0"
description = "A computer vision project"
authors = ["Enrique G. Ortiz <ortizeg@gmail.com>"]
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"          # system library -- always from conda

[pypi-dependencies]
torch = ">=2.2"           # PyTorch ecosystem -- from PyPI for reliable CUDA
lightning = ">=2.2"
pydantic = ">=2.6"

[feature.dev.dependencies]
pytest = ">=7.4"
ruff = ">=0.8"
mypy = ">=1.11"

[environments]
default = { features = ["dev"], solve-group = "default" }

[tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
quality = { depends-on = ["lint", "format-check", "typecheck"] }
```

## The command core

```bash
pixi install                       # install default env (creates/updates pixi.lock)
pixi install -e train              # install a named environment

pixi add numpy ">=1.26"            # add a conda dependency
pixi add --pypi torch ">=2.2"      # add a PyPI dependency
pixi add --feature dev pytest ">=7.4"   # add to a feature
pixi remove numpy

pixi run test                      # run a task
pixi run -e train train            # run a task in a named environment
pixi shell                         # drop into the environment
pixi list                          # list installed packages
pixi info                          # show environment info
```

## Features and environments at a glance

Keep the default environment lean by putting optional dependencies behind named features,
then compose features into environments that share a `solve-group`.

```toml
[feature.cuda.dependencies]
cuda-toolkit = ">=12.1"

[feature.docs.dependencies]
mkdocs = ">=1.5"

[environments]
default = { features = ["dev"], solve-group = "default" }
train = { features = ["dev", "cuda"], solve-group = "default" }
docs = { features = ["dev", "docs"], solve-group = "default" }
```

## Choosing conda vs PyPI

1. **System libraries** (OpenCV, FFmpeg, CUDA): always use conda.
2. **PyTorch ecosystem** (torch, torchvision, torchaudio): use PyPI for reliable CUDA support.
3. **Pure Python packages** (pydantic, wandb, lightning): either works; prefer PyPI for latest versions.
4. **Packages with C extensions** (Pillow, numpy, scipy): conda-forge often provides better-optimized builds.

## Conventions

- Pin **minimum** versions in `pixi.toml` (`>=1.26`); let `pixi.lock` pin exact versions.
- Commit `pixi.lock` for **applications** (training projects, services); gitignore it for **libraries**.
- Keep the default environment lean — put optional dependencies behind named features (`dev`, `docs`, `cuda`, `onnx`, `wandb`).
- Use a shared `solve-group` across environments so features resolve to consistent versions.
- Define every repeatable command as a task, and compose them with `depends-on` instead of shell scripts.
- Always declare all target platforms up front (`linux-64`, `osx-arm64`, …) so the lock file covers every machine.

## Anti-patterns to avoid

1. **Never use `pip install` directly** -- always use `pixi add --pypi`.
2. **Never use `conda install`** -- use `pixi add`.
3. **Never create requirements.txt** -- pixi.toml and pixi.lock replace it.
4. **Never create environment.yml** -- pixi.toml replaces it.
5. **Never use virtualenv or venv** -- pixi manages environments automatically.
6. **Never pin exact versions in pixi.toml** -- use minimum versions and let the lock file handle exact pinning.
7. **Never mix conda and pip channels for the same package** -- choose one source per package.
8. **Never forget to commit pixi.lock for applications** -- it ensures reproducibility.

## Deep dives

- `references/manifest-reference.md` — read when authoring a full `pixi.toml` from scratch or moving the pixi config into `pyproject.toml` under `[tool.pixi.*]`.
- `references/dependencies.md` — read when deciding conda vs PyPI for a package, adding CUDA/system libraries, or choosing a version-pinning strategy.
- `references/environments-and-features.md` — read when adding optional dependency groups (dev, docs, cuda, onnx, wandb) or composing them into named environments.
- `references/tasks.md` — read when defining tasks with dependencies, environment variables, or feature-scoped tasks.
- `references/cli-and-lockfiles.md` — read when you need the full command reference or lock-file/gitignore policy for applications vs libraries.
- `references/project-templates.md` — read when bootstrapping a training project, an inference service, or a library package and you want a complete known-good manifest.

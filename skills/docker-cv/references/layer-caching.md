# Layer Caching and Image Size

Scope: ordering Dockerfile layers for maximum cache reuse, the `.dockerignore` baseline
for ML repos, and the layer habits that keep CV images from bloating.

## Contents

- [Layer Ordering for Cache Optimization](#layer-ordering-for-cache-optimization)
- [.dockerignore](#dockerignore)
- [Large Image Sizes](#large-image-sizes)
- [Inspecting Layer Sizes](#inspecting-layer-sizes)

## Layer Ordering for Cache Optimization

Order your Dockerfile layers from least-frequently-changed to most-frequently-changed:

```
1. Base image + system packages     (rarely changes)
2. Package manager install          (rarely changes)
3. Dependency files (pixi.toml)     (changes with new deps)
4. pip install / pixi install       (depends on step 3)
5. pyproject.toml                   (changes occasionally)
6. Source code (src/)               (changes every commit)
7. Configs and scripts              (changes frequently)
```

This maximizes Docker's layer cache reuse. When only source code changes, steps 1-4 are
cached.

The practical consequence is that dependency manifests are copied on their own, before
the source tree:

```dockerfile
# Copy only dependency files first (cache optimization)
COPY pixi.toml pixi.lock ./
RUN pixi install

# Source code last — changes on every commit
COPY src/ src/
```

A `COPY . .` before `RUN pixi install` invalidates the dependency layer on every source
edit, forcing a full reinstall on each build.

## .dockerignore

```
# .dockerignore for ML projects
.git/
.github/
.vscode/
.mypy_cache/
.pytest_cache/
.ruff_cache/
__pycache__/
*.pyc

# Data and artifacts (mount as volumes instead)
data/
checkpoints/
outputs/
wandb/
mlruns/
lightning_logs/

# Large model files (copy explicitly if needed)
*.pt
*.pth
*.onnx
*.pkl

# Documentation and tests
docs/
tests/
*.md
!README.md

# OS files
.DS_Store
Thumbs.db

# Environment
.env
.venv/
.pixi/
```

Without a `.dockerignore`, the build context includes `data/`, `checkpoints/`, and
`.git/` — often tens of gigabytes shipped to the daemon on every build.

## Large Image Sizes

```dockerfile
# ✅ Combine RUN commands to reduce layers
RUN apt-get update && \
    apt-get install -y --no-install-recommends pkg1 pkg2 && \
    rm -rf /var/lib/apt/lists/*

# ❌ Separate RUN creates extra layers
RUN apt-get update
RUN apt-get install -y pkg1
RUN apt-get install -y pkg2
```

Cleanup must happen *in the same layer* as the install. `rm -rf /var/lib/apt/lists/*` in a
later `RUN` does not shrink the image, because the earlier layer still contains the files.
The same applies to `pip install --no-cache-dir` versus deleting `~/.cache/pip` afterwards.

## Inspecting Layer Sizes

```bash
# Per-layer size breakdown
docker history myproject:latest

# Total image size comparison
docker images myproject
```

Use this to find the layer responsible for a size regression before guessing at fixes.

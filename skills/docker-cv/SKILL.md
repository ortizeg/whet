---
name: docker-cv
description: >
  Use this skill when writing or optimizing a Dockerfile for computer vision or deep
  learning — CUDA/GPU base images, multi-stage builds, layer caching, slim inference
  images, non-root security, and shrinking bloated CV images. Reach for it any time
  you'd otherwise hand-write a GPU Dockerfile or debug a container that won't see the
  GPU, even if the user just says "containerize this model". For deploying the resulting
  containers to a cluster see kubernetes; for cloud image registries see gcp and
  aws-sagemaker.
---

# Docker CV Skill

Build optimized Docker images for computer vision and deep learning workloads with CUDA
support, multi-stage builds, and security best practices. This page holds the canonical
pixi + CUDA Dockerfile that archetype templates follow; the deep dives cover GPU
specifics, caching, hardening, and Compose.

## Choosing a containerization approach

```
Need to package an ML application?
├── Single model serving → Dockerfile with multi-stage build
├── Multiple models/services → Docker Compose for local, K8s for prod
├── GPU required?
│   ├── Training → NVIDIA base images (nvcr.io/nvidia/pytorch)
│   └── Inference → optimized runtime images (NVIDIA Triton, TorchServe)
└── No GPU → Python slim base image
```

## The canonical multi-stage pixi Dockerfile

Base stage for CUDA and system libraries, dependencies stage for pixi, leaf stages for
training and inference. Dependency files are copied before source so `pixi install` stays
cached.

```dockerfile
# ==============================================================================
# Stage 1: Base — CUDA runtime + system dependencies
# ==============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Dependencies — install Python packages
# ==============================================================================
FROM base AS dependencies

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

WORKDIR /app

# Copy only dependency files first (cache optimization)
COPY pixi.toml pixi.lock ./

# Install dependencies (cached unless pixi.toml/lock changes)
RUN pixi install

# ==============================================================================
# Stage 3: Training — full development image
# ==============================================================================
FROM dependencies AS training

# Copy source code (changes frequently, so last)
COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

# Install project in dev mode
RUN pixi run pip install -e ".[dev]"

# Non-root user for security
RUN useradd -m -u 1000 trainer
USER trainer

ENTRYPOINT ["pixi", "run", "python", "-m"]
CMD ["my_project.train"]
```

Build a specific stage with `docker build --target training -t myproject:train .`.
The slim inference stage lives in `references/multi-stage-builds.md`.

## Base image selection

| Use Case | Base Image | Size |
|----------|-----------|------|
| Training (GPU) | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | ~3.5 GB |
| Training (CPU) | `python:3.11-slim` | ~150 MB |
| Inference (GPU) | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | ~2.8 GB |
| Inference (CPU) | `python:3.11-slim` | ~150 MB |
| Development | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | ~5.2 GB |

## Conventions

1. **Always use multi-stage builds** -- separate training from inference
2. **Pin base image versions** -- never use `latest` in production
3. **Use `.dockerignore`** -- exclude data, checkpoints, and dev files
4. **Run as non-root** -- create a dedicated user
5. **No secrets in images** -- use environment variables or secrets managers
6. **Optimize layer order** -- dependencies before source code
7. **Set `shm_size`** -- prevent DataLoader shared memory crashes
8. **Add health checks** -- especially for inference containers
9. **Use `--no-install-recommends`** -- minimize system package installs
10. **Clean up in the same layer** -- `rm -rf /var/lib/apt/lists/*` after `apt-get`

## Anti-patterns

- **`COPY . .` before installing dependencies** — invalidates the dependency layer on
  every source edit; copy `pixi.toml`/`pixi.lock` first, then run `pixi install`.
- **Default shared memory** — the 64 MB default kills `DataLoader` workers; set
  `shm_size: "8gb"` (or `--shm-size=8g`).
- **Secrets in `ENV` or a copied `.env`** — every layer is retained and readable with
  `docker history`; pass secrets at runtime instead.
- **`devel` CUDA images in production** — the CUDA SDK adds ~1.7 GB and is build-time
  only; ship a `runtime` variant.
- **Cleanup in a separate `RUN`** — `rm -rf /var/lib/apt/lists/*` only shrinks the image
  when it runs in the same layer as `apt-get install`.
- **Baking datasets or checkpoints into the image** — mount them as volumes.
- **Running as root** — create and switch to a non-root user before the entrypoint.
- **Unpinned CUDA tags** — pin the full version and document the minimum NVIDIA driver.

## Deep dives

- `references/multi-stage-builds.md` — read when writing the full Dockerfile, adding the slim inference stage, or deciding which stages a workload needs.
- `references/cuda-and-gpu.md` — read when picking a CUDA tag, hitting a driver/CUDA mismatch, a container that can't see the GPU, or DataLoader shared-memory crashes.
- `references/layer-caching.md` — read when build times or image size need optimizing, or when writing `.dockerignore`.
- `references/security-and-slim-images.md` — read when hardening a container: non-root users, secret handling, read-only filesystems, health checks, and trimming the serving image.
- `references/compose.md` — read when standing up a local multi-service stack (training + TensorBoard + inference) with GPU reservations.

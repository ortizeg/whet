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

Build optimized Docker images for computer vision and deep learning workloads with CUDA support, multi-stage builds, and security best practices.

## Multi-Stage Build Strategy

Use separate stages to minimize final image size and maximize layer cache reuse.

### Training Dockerfile

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

### Inference Dockerfile

```dockerfile
# ==============================================================================
# Slim inference image — no CUDA SDK, just runtime
# ==============================================================================
FROM python:3.11-slim AS inference

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install only production dependencies
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy application code and model
COPY src/ src/
COPY models/ models/

# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Health check for inference service
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Docker Compose

### Training with GPU

```yaml
# docker-compose.yml
services:
  train:
    build:
      context: .
      dockerfile: Dockerfile
      target: training
    volumes:
      - ./data:/app/data:ro          # Read-only data mount
      - ./checkpoints:/app/checkpoints  # Writable checkpoint output
      - ./configs:/app/configs:ro    # Config overrides
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0,1
    shm_size: "8gb"  # Required for DataLoader num_workers > 0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["my_project.train", "experiment=baseline"]

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./outputs/logs:/logs:ro
    command: ["tensorboard", "--logdir=/logs", "--bind_all"]

  inference:
    build:
      context: .
      dockerfile: Dockerfile
      target: inference
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

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

This maximizes Docker's layer cache reuse. When only source code changes, steps 1-4 are cached.

## Base Image Selection

| Use Case | Base Image | Size |
|----------|-----------|------|
| Training (GPU) | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | ~3.5 GB |
| Training (CPU) | `python:3.11-slim` | ~150 MB |
| Inference (GPU) | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | ~2.8 GB |
| Inference (CPU) | `python:3.11-slim` | ~150 MB |
| Development | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | ~5.2 GB |

## Security Best Practices

### Non-Root User

```dockerfile
# Always create and switch to a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

### No Secrets in Images

```dockerfile
# ❌ WRONG: Secret baked into image layer
ENV WANDB_API_KEY=my-secret-key
COPY .env /app/.env

# ✅ CORRECT: Pass at runtime
# docker run -e WANDB_API_KEY=$WANDB_API_KEY my-image
# docker-compose with env_file or environment
```

### Read-Only Filesystem

```yaml
# docker-compose.yml
services:
  inference:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - ./models:/app/models:ro
```

## Multi-Platform Builds

```bash
# Build for both AMD64 and ARM64
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag myproject:latest \
    --push .
```

## Health Checks

```dockerfile
# HTTP health check for API services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# File-based health check for training containers
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD test -f /tmp/training_alive || exit 1
```

## Common Issues

### DataLoader Crashes

If `DataLoader` with `num_workers > 0` crashes with shared memory errors:

```yaml
# Increase shared memory size
services:
  train:
    shm_size: "8gb"  # Default is 64MB, way too small
```

### CUDA Version Mismatch

Always pin CUDA versions and document driver requirements:

```dockerfile
# Pin specific CUDA version
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Document minimum driver version in README
# Requires NVIDIA driver >= 550.54.15
```

### Large Image Sizes

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

## Best Practices

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

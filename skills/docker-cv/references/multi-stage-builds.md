# Multi-Stage Docker Builds for CV/ML

Scope: the full multi-stage build strategy for CV workloads — the complete training
Dockerfile (CUDA + pixi), the slim inference Dockerfile, and how the stages relate.

## Contents

- [Multi-Stage Build Strategy](#multi-stage-build-strategy)
- [Training Dockerfile](#training-dockerfile)
- [Inference Dockerfile](#inference-dockerfile)
- [Choosing Stages per Workload](#choosing-stages-per-workload)

## Multi-Stage Build Strategy

Use separate stages to minimize final image size and maximize layer cache reuse. A single
Dockerfile defines a `base` stage (CUDA runtime + system libraries), a `dependencies`
stage (package manager + locked dependencies), and one or more leaf stages (`training`,
`inference`) that are selected at build time with `--target`.

```bash
docker build --target training --tag myproject:train .
docker build --target inference --tag myproject:serve .
```

## Training Dockerfile

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

The system libraries in the base stage are the ones OpenCV and video decoding need:
`libgl1-mesa-glx`, `libglib2.0-0`, `libsm6`, `libxext6`, `libxrender-dev`, and `ffmpeg`.
Missing them is the usual cause of `ImportError: libGL.so.1: cannot open shared object
file` inside a container.

## Inference Dockerfile

The inference image drops the CUDA SDK and pixi entirely when the model runs on CPU or on
a runtime-only CUDA base, installing just the production dependency set.

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

## Choosing Stages per Workload

```
Need to package an ML application?
├── Single model serving → Dockerfile with multi-stage build
├── Multiple models/services → Docker Compose for local, K8s for prod
├── GPU required?
│   ├── Training → NVIDIA base images (nvcr.io/nvidia/pytorch)
│   └── Inference → optimized runtime images (NVIDIA Triton, TorchServe)
└── No GPU → Python slim base image
```

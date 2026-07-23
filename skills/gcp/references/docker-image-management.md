# Docker Image Management for GCP

Scope: building, tagging, and pushing training/inference images to Artifact Registry for Vertex AI and GKE workloads, including CI.

## Contents

- [Build and Push Workflow](#build-and-push-workflow)
- [Multi-Stage for Training and Inference](#multi-stage-for-training-and-inference)
- [GitHub Actions: Build and Push to Artifact Registry](#github-actions-build-and-push-to-artifact-registry)

## Build and Push Workflow

```bash
# Variables
PROJECT_ID="my-project"
REGION="us-central1"
REPO="ml-images"
IMAGE="training"
TAG="v1.2.0"
FULL_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

# Build with cache
docker build \
    --tag "${FULL_URI}" \
    --cache-from "${FULL_URI}" \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -f Dockerfile.training .

# Push
docker push "${FULL_URI}"
```

## Multi-Stage for Training and Inference

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

# Training — full environment with dev tools
FROM base AS training
WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install
COPY src/ src/
COPY configs/ configs/
RUN useradd -m -u 1000 trainer
USER trainer
ENTRYPOINT ["pixi", "run", "python", "-m"]
CMD ["my_project.train"]

# Inference — minimal runtime
FROM base AS inference
WORKDIR /app
COPY pixi.toml pixi.lock ./
RUN pixi install
COPY src/ src/
RUN useradd -m -u 1000 appuser
USER appuser
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["pixi", "run", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

## GitHub Actions: Build and Push to Artifact Registry

```yaml
# .github/workflows/docker-gcp.yml
name: Build and Push to Artifact Registry

on:
  push:
    tags: ["v*"]

env:
  PROJECT_ID: my-project
  REGION: us-central1
  REPOSITORY: ml-images

jobs:
  build-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write

    steps:
      - uses: actions/checkout@v4

      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}

      - uses: google-github-actions/setup-gcloud@v2

      - run: gcloud auth configure-docker ${{ env.REGION }}-docker.pkg.dev

      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/training:${{ github.ref_name }}
            ${{ env.REGION }}-docker.pkg.dev/${{ env.PROJECT_ID }}/${{ env.REPOSITORY }}/training:latest
          target: training
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

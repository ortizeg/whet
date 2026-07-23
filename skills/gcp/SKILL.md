---
name: gcp
description: >
  Use this skill when working with Google Cloud infrastructure for CV/ML — pushing
  images to Artifact Registry, storing datasets and checkpoints in Cloud Storage (gsutil,
  gcsfuse, the Python client), and provisioning with the gcloud CLI. Reach for it any
  time the project touches GCP buckets, registries, or gcloud commands, even if the user
  doesn't name the specific service — including submitting Vertex AI custom training jobs
  and retrieving their artifacts. For the AWS equivalent see aws-sagemaker.
---

# GCP Skill

Google Cloud Platform services for CV/ML projects: Artifact Registry, Cloud Storage, Vertex AI training, and Docker image management.

## Artifact Registry

Use Artifact Registry (replaces the deprecated Container Registry) for Docker images and Python packages.

### Create a Docker Repository

```bash
# Create a Docker repository in Artifact Registry
gcloud artifacts repositories create ml-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV/ML training and inference images"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

### Push and Pull Docker Images

```bash
# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Tag and push
docker tag my-training:latest \
    us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

docker push us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

# Pull
docker pull us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
```

### Python Package Repository

```bash
gcloud artifacts repositories create ml-packages \
    --repository-format=python --location=us-central1

# Print pip/uv index settings for the repo
gcloud artifacts print-settings python --repository=ml-packages --location=us-central1
```

Add the printed index to `pyproject.toml` under `[tool.uv]` as an `extra-index-url` (e.g. `https://us-central1-python.pkg.dev/my-project/ml-packages/simple/`).

## Cloud Storage

Use Cloud Storage for datasets, model checkpoints, and training artifacts.

### Upload and Download with gsutil

```bash
# Upload a dataset
gsutil -m cp -r ./data/coco/ gs://my-ml-bucket/datasets/coco/

# Download model checkpoint
gsutil cp gs://my-ml-bucket/checkpoints/resnet50_epoch20.pt ./checkpoints/

# Sync outputs (only transfers changed files)
gsutil -m rsync -r ./outputs/ gs://my-ml-bucket/runs/experiment-42/
```

### Python Client

```python
from google.cloud import storage


def upload_model_artifact(
    bucket_name: str,
    source_path: str,
    destination_blob: str,
) -> str:
    """Upload a model artifact to Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket_name}/{destination_blob}"


def download_checkpoint(
    bucket_name: str,
    blob_name: str,
    destination_path: str,
) -> None:
    """Download a checkpoint from Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
```

### Mount with gcsfuse

```bash
# Mount a bucket as a local directory (useful for large datasets)
gcsfuse --implicit-dirs my-ml-bucket /mnt/gcs-data

# Use in a training container
docker run --privileged \
    -v /mnt/gcs-data:/data:ro \
    my-training:latest
```

```dockerfile
# Install gcsfuse in a training container
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-jammy main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    rm -rf /var/lib/apt/lists/*
```

## Vertex AI Training Jobs

Submit custom training jobs to Vertex AI for GPU-accelerated model training.

### Custom Training Job

```python
from google.cloud import aiplatform


def submit_training_job(
    project: str,
    location: str,
    display_name: str,
    container_uri: str,
    args: list[str],
    machine_type: str = "n1-standard-8",
    accelerator_type: str = "NVIDIA_TESLA_T4",
    accelerator_count: int = 1,
    staging_bucket: str = "",
) -> aiplatform.CustomJob:
    """Submit a custom training job to Vertex AI."""
    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomJob.from_local_script(
        display_name=display_name,
        script_path="src/train.py",
        container_uri=container_uri,
        args=args,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
    )
    job.run(sync=False)
    return job
```

### Custom Container Training

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomContainerTrainingJob(
    display_name="resnet50-finetune",
    container_uri="us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0",
    command=["python", "-m", "my_project.train"],
    model_serving_container_image_uri=(
        "us-central1-docker.pkg.dev/my-project/ml-images/inference:v1.2.0"
    ),
)

model = job.run(
    args=["--config=configs/finetune.yaml", "--epochs=50"],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count=2,
    base_output_dir="gs://my-ml-bucket/vertex-outputs/",
)
```

### GPU Selection Reference

| GPU Type | Vertex AI Name | Use Case |
|----------|----------------|----------|
| T4 | `NVIDIA_TESLA_T4` | Inference, small-batch training |
| V100 | `NVIDIA_TESLA_V100` | Training (16 GB HBM2) |
| A100 40GB | `NVIDIA_TESLA_A100` | Large-scale training |
| A100 80GB | `NVIDIA_A100_80GB` | Large models, large batch sizes |
| L4 | `NVIDIA_L4` | Inference, cost-effective training |
| H100 | `NVIDIA_H100_80GB` | Highest throughput training |

### Prebuilt Training Containers

Skip building a custom image: pass a Google prebuilt container (e.g. `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest`) as `container_uri` to `from_local_script`, plus `requirements=["torchvision", "albumentations", ...]` for extra deps.

### Retrieving Artifacts After Training

**The training VM is ephemeral — anything not written to GCS is gone when the job ends.**
Write checkpoints and exports to the GCS path Vertex provides (`AIP_MODEL_DIR`), then pull
them down explicitly once the job finishes. Do not assume the job "left them somewhere".

```python
import os

# Inside the training container: write to the GCS path Vertex provides.
# Falls back to a local dir so the same script runs off-cloud unchanged.
model_dir = os.environ.get("AIP_MODEL_DIR", "outputs/model")
trainer.save_checkpoint(f"{model_dir}/best.ckpt")
```

```bash
# After the job completes: sync every artifact to the local machine.
gcloud storage rsync -r "gs://my-bucket/jobs/${JOB_ID}/model" ./artifacts/

# Verify before deleting anything remote.
ls -lh ./artifacts/
```

## Docker Image Management

Build, tag, and push images to Artifact Registry for Vertex AI and GKE workloads.

### Build and Push Workflow

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

### Multi-Stage for Training and Inference

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

### GitHub Actions: Build and Push to Artifact Registry

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

## Authentication

### Service Account Setup

```bash
# Create a service account for training jobs
gcloud iam service-accounts create ml-trainer \
    --display-name="ML Training Service Account"

# Grant required roles (least privilege)
SA_EMAIL="ml-trainer@my-project.iam.gserviceaccount.com"
for ROLE in aiplatform.user storage.objectAdmin artifactregistry.reader; do
    gcloud projects add-iam-policy-binding my-project \
        --member="serviceAccount:${SA_EMAIL}" --role="roles/${ROLE}"
done
```

### Workload Identity Federation for CI

```bash
# Create a workload identity pool for GitHub Actions
gcloud iam workload-identity-pools create "github-pool" \
    --location="global" \
    --display-name="GitHub Actions Pool"

gcloud iam workload-identity-pools providers create-oidc "github-provider" \
    --location="global" \
    --workload-identity-pool="github-pool" \
    --display-name="GitHub Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --issuer-uri="https://token.actions.githubusercontent.com"

# Allow the GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/OWNER/REPO"
```

### Docker Authentication

```bash
# Local: gcloud auth configure-docker us-central1-docker.pkg.dev
# CI (prefer WIF over keys): cat key.json | docker login -u _json_key \
#   --password-stdin https://us-central1-docker.pkg.dev
```

## Pydantic Configuration

Typed models for GCP project settings and job specs.

```python
from pydantic import BaseModel, Field


class GCPConfig(BaseModel, frozen=True):
    project_id: str = Field(description="GCP project ID")
    region: str = Field(default="us-central1")
    zone: str = Field(default="us-central1-a")


class ArtifactRegistryConfig(BaseModel, frozen=True):
    """Artifact Registry repository settings."""

    repository: str = Field(description="Repository name")
    location: str = Field(default="us-central1")

    def docker_uri(self, project_id: str, image: str, tag: str) -> str:
        """Build the full Docker image URI."""
        return (
            f"{self.location}-docker.pkg.dev/{project_id}"
            f"/{self.repository}/{image}:{tag}"
        )


class StorageBucketConfig(BaseModel, frozen=True):
    """Cloud Storage bucket configuration."""

    bucket_name: str = Field(description="GCS bucket name")
    datasets_prefix: str = Field(default="datasets/")
    checkpoints_prefix: str = Field(default="checkpoints/")
    outputs_prefix: str = Field(default="outputs/")

    def dataset_uri(self, name: str) -> str:
        return f"gs://{self.bucket_name}/{self.datasets_prefix}{name}"

    def checkpoint_uri(self, name: str) -> str:
        return f"gs://{self.bucket_name}/{self.checkpoints_prefix}{name}"


class VertexJobConfig(BaseModel, frozen=True):
    """Vertex AI training job configuration."""

    display_name: str
    machine_type: str = Field(default="n1-standard-8")
    accelerator_type: str = Field(default="NVIDIA_TESLA_T4")
    accelerator_count: int = Field(default=1, ge=1, le=8)
    replica_count: int = Field(default=1, ge=1)
    staging_bucket: str = Field(description="GCS bucket for staging")
    boot_disk_size_gb: int = Field(default=100, ge=50)


class GCPProjectConfig(BaseModel, frozen=True):
    """Complete GCP configuration — composed from a Hydra-compatible configs/gcp.yaml."""

    gcp: GCPConfig
    artifact_registry: ArtifactRegistryConfig
    storage: StorageBucketConfig
    vertex_job: VertexJobConfig
```

## Project Dependencies

Add the GCP client libraries with `pixi add google-cloud-storage google-cloud-aiplatform` (and `google-cloud-artifact-registry` as a dev dependency). Wrap the recurring `gcloud`/`gsutil` commands above in a `justfile` for team consistency — e.g. `just gcs-sync-outputs`, `just docker-push-train`, `just vertex-list-jobs` (`gcloud ai custom-jobs list --region=us-central1`).

## Best Practices

1. **Artifact Registry, not Container Registry** — the latter is deprecated.
2. **Pin image tags for Vertex AI jobs** — semantic versions or Git SHAs, never `:latest` in production.
3. **Use Workload Identity Federation** — OIDC tokens over long-lived service account keys.
4. **Keep datasets in Cloud Storage, not images** — mount via gcsfuse or download at job start.
5. **Set `staging_bucket`** — Vertex AI needs one for scripts and intermediate artifacts.
6. **Write outputs to `AIP_MODEL_DIR` and pull them down when the job finishes** — the training VM is ephemeral; un-synced artifacts are lost.
7. **Keep resources regional and co-located** to minimize egress cost and latency.
8. **Configure GCS lifecycle rules** to auto-delete stale checkpoints/outputs.
9. **Prefer prebuilt Vertex AI containers** — optimized CUDA/NCCL.
10. **Tag with both version and `latest`** — reproducibility plus dev convenience.
11. **Grant least-privilege IAM** — `roles/aiplatform.user` for jobs, `roles/storage.objectViewer` for read-only data.

## Anti-Patterns

- ❌ Using Container Registry (`gcr.io/`) for new projects — use `pkg.dev/`.
- ❌ Baking credentials/SA keys into images — use env vars or Workload Identity.
- ❌ Running Vertex jobs as the default Compute Engine SA — use a dedicated minimal-permission SA.
- ❌ Storing datasets inside images — huge, slow to pull; mount from GCS.
- ❌ `gcloud auth print-access-token` in scripts — tokens expire hourly; use ADC or SA impersonation.
- ❌ Submitting Vertex jobs without a staging bucket.
- ❌ Multi-region buckets for single-region training data — extra egress, no benefit.
- ❌ Hardcoding project IDs/regions — use Pydantic config or env vars.

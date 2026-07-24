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

Google Cloud Platform services for CV/ML projects: Artifact Registry for Docker images and
Python packages, Cloud Storage for datasets and checkpoints, Vertex AI for GPU training jobs,
and the gcloud CLI for everything else. Add clients with
`pixi add google-cloud-storage google-cloud-aiplatform`.

## Essential Core

The two operations that come up constantly: push an image to Artifact Registry, and move data
to/from a GCS bucket.

```bash
# Push a training image to Artifact Registry (gcr.io is deprecated — use pkg.dev)
gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag my-training:latest \
    us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
docker push us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
```

```bash
# Datasets up, checkpoints down, outputs synced
gsutil -m cp -r ./data/coco/ gs://my-ml-bucket/datasets/coco/
gsutil cp gs://my-ml-bucket/checkpoints/resnet50_epoch20.pt ./checkpoints/
gsutil -m rsync -r ./outputs/ gs://my-ml-bucket/runs/experiment-42/
```

## Vertex AI Artifacts Are Not Automatic

**The Vertex training VM is ephemeral — anything not written to GCS is gone when the job ends.**
Write checkpoints and exports to the GCS path Vertex provides (`AIP_MODEL_DIR`), then pull them
down explicitly once the job finishes. Do not assume the job "left them somewhere".

```python
import os

model_dir = os.environ.get("AIP_MODEL_DIR", "outputs/model")  # local fallback off-cloud
trainer.save_checkpoint(f"{model_dir}/best.ckpt")
```

```bash
gcloud storage rsync -r "gs://my-bucket/jobs/${JOB_ID}/model" ./artifacts/
ls -lh ./artifacts/   # verify before deleting anything remote
```

Full job-submission detail is in `references/vertex-training.md`.

## Everyday gcloud Commands

```bash
# Auth: prefer Application Default Credentials over key files
gcloud auth application-default login
gcloud config set project my-project
gcloud config set compute/region us-central1

# Repositories and images
gcloud artifacts repositories list --location=us-central1
gcloud artifacts docker images list us-central1-docker.pkg.dev/my-project/ml-images

# Vertex AI jobs
gcloud ai custom-jobs list --region=us-central1
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# Buckets
gcloud storage ls gs://my-ml-bucket/
gcloud storage rsync -r gs://my-ml-bucket/runs/experiment-42/ ./outputs/
```

Wrap the recurring ones in a `justfile` (`just gcs-sync-outputs`, `just docker-push-train`,
`just vertex-list-jobs`) so the team runs identical commands.

## Conventions

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

## Deep dives

- `references/artifact-registry.md` — read when creating Docker or Python package repositories, or wiring a private index into `pyproject.toml`.
- `references/cloud-storage.md` — read when using the Python `google.cloud.storage` client or mounting a bucket with gcsfuse.
- `references/vertex-training.md` — read when submitting a custom training job, choosing a GPU/accelerator, using prebuilt containers, or retrieving artifacts after a run.
- `references/docker-image-management.md` — read when writing multi-stage training/inference Dockerfiles or a GitHub Actions build-and-push workflow.
- `references/authentication.md` — read when setting up service accounts, IAM roles, or Workload Identity Federation for CI.
- `references/pydantic-configuration.md` — read when modelling GCP project/bucket/job settings as typed config or wiring `just` targets.

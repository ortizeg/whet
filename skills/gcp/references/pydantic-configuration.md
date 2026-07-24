# Typed GCP Configuration and Project Dependencies

Scope: Pydantic models for GCP project settings, registry URIs, buckets, and Vertex job specs — plus the client libraries and task-runner wiring that go with them.

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

Add the GCP client libraries with `pixi add google-cloud-storage google-cloud-aiplatform` (and `google-cloud-artifact-registry` as a dev dependency). Wrap the recurring `gcloud`/`gsutil` commands in a `justfile` for team consistency — e.g. `just gcs-sync-outputs`, `just docker-push-train`, `just vertex-list-jobs` (`gcloud ai custom-jobs list --region=us-central1`).

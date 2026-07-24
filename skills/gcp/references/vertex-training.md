# Vertex AI Training Jobs

Scope: submitting custom training jobs to Vertex AI, choosing accelerators, using prebuilt containers, and retrieving artifacts from the ephemeral training VM.

## Contents

- [Custom Training Job](#custom-training-job)
- [Custom Container Training](#custom-container-training)
- [GPU Selection Reference](#gpu-selection-reference)
- [Prebuilt Training Containers](#prebuilt-training-containers)
- [Retrieving Artifacts After Training](#retrieving-artifacts-after-training)

Submit custom training jobs to Vertex AI for GPU-accelerated model training.

## Custom Training Job

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

## Custom Container Training

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

## GPU Selection Reference

| GPU Type | Vertex AI Name | Use Case |
|----------|----------------|----------|
| T4 | `NVIDIA_TESLA_T4` | Inference, small-batch training |
| V100 | `NVIDIA_TESLA_V100` | Training (16 GB HBM2) |
| A100 40GB | `NVIDIA_TESLA_A100` | Large-scale training |
| A100 80GB | `NVIDIA_A100_80GB` | Large models, large batch sizes |
| L4 | `NVIDIA_L4` | Inference, cost-effective training |
| H100 | `NVIDIA_H100_80GB` | Highest throughput training |

## Prebuilt Training Containers

Skip building a custom image: pass a Google prebuilt container (e.g. `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest`) as `container_uri` to `from_local_script`, plus `requirements=["torchvision", "albumentations", ...]` for extra deps.

## Retrieving Artifacts After Training

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

Related job management: `gcloud ai custom-jobs list --region=us-central1` lists submitted jobs;
wrap it in a `justfile` target such as `just vertex-list-jobs`.

---
name: vertex-ai
description: >
  Use this skill to run CV/ML training on Google Cloud Vertex AI — whenever you need to
  submit a custom-container training job, launch a GPU run in the cloud, stage data and
  artifacts in GCS, tune hyperparameters, or orchestrate a Vertex Pipeline. Reach for it
  any time training moves off the laptop onto Vertex. Covers the aiplatform Python SDK,
  custom containers from Artifact Registry, accelerator config, and W&B integration.
---

# Vertex AI (Custom Training)

The pattern here is **custom-container training**: package the training code into a Docker
image, push it to Artifact Registry, and submit it to Vertex AI as a CustomJob that runs
on a GPU machine, reading data from and writing checkpoints back to GCS. This keeps the
exact same container running locally and in the cloud — no Vertex-specific training code.

Pairs with the `gcp` skill (Artifact Registry, GCS, auth), `docker-cv` (the image), and
`wandb` (tracking). Use the `google-cloud-aiplatform` SDK.

## Initialize

```python
from __future__ import annotations

from google.cloud import aiplatform

aiplatform.init(
    project="my-gcp-project",
    location="us-central1",
    staging_bucket="gs://my-project-vertex/staging",
)
```

## Submit a custom-container training job

The training image is your normal `docker-cv` image with an entrypoint that trains and
writes to the GCS path Vertex provides via `AIP_MODEL_DIR`.

```python
job = aiplatform.CustomContainerTrainingJob(
    display_name="rfdetr-finetune",
    container_uri="us-central1-docker.pkg.dev/my-gcp-project/train/rfdetr:latest",
    staging_bucket="gs://my-project-vertex/staging",
)

model = job.run(
    args=[
        "--config", "configs/rfdetr.yaml",
        "--data", "gs://my-project-data/coco-subset",
        "--epochs", "50",
    ],
    replica_count=1,
    machine_type="a2-highgpu-1g",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
    environment_variables={"WANDB_API_KEY": WANDB_KEY, "WANDB_PROJECT": "rfdetr"},
    boot_disk_size_gb=200,
    sync=True,
)
```

For full control (no managed model artifact), use `CustomJob.from_local_script` or a raw
`CustomJob` with a `worker_pool_specs` list — same machine/accelerator fields.

## The training entrypoint (GCS-aware)

Inside the container, read Vertex's env vars so the same script runs locally and on Vertex.

```python
import os

from loguru import logger

# Vertex sets these; fall back to local paths when running off-cloud.
model_dir = os.environ.get("AIP_MODEL_DIR", "outputs/model")
logger.info("writing checkpoints to {}", model_dir)

# torch/gcsfs can write directly to gs:// paths; or stage locally then upload.
trainer.fit(...)
trainer.save_checkpoint(f"{model_dir}/best.ckpt")
```

## Data and artifacts in GCS

- **Read** datasets straight from `gs://` with `gcsfs`/`fsspec`, or copy to the local SSD
  first for many-small-file IO (detection datasets): `gcloud storage cp -r gs://... /data`.
- **Write** checkpoints/exports to `AIP_MODEL_DIR` (a GCS path) so they survive the
  ephemeral training VM.

```python
import gcsfs

fs = gcsfs.GCSFileSystem(project="my-gcp-project")
with fs.open("gs://my-project-data/labels.json") as f:
    labels = f.read()
```

## Hyperparameter tuning

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

tuning = aiplatform.HyperparameterTuningJob(
    display_name="rfdetr-hpt",
    custom_job=job,  # a CustomJob whose entrypoint reports a metric
    metric_spec={"val_map": "maximize"},
    parameter_spec={
        "lr": hpt.DoubleParameterSpec(min=1e-5, max=1e-3, scale="log"),
        "weight_decay": hpt.DoubleParameterSpec(min=1e-5, max=1e-2, scale="log"),
    },
    max_trial_count=20,
    parallel_trial_count=4,
)
tuning.run()
```

The training script reports the objective back to Vertex with `hypertune`:

```python
import hypertune

hpt_client = hypertune.HyperTune()
hpt_client.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="val_map", metric_value=val_map, global_step=epoch,
)
```

## Vertex Pipelines (multi-step)

For a full download → train → evaluate → register flow, define a KFP pipeline of
containerized components and submit it. Keep components thin wrappers over the same
training image.

```python
from kfp import dsl
from google.cloud import aiplatform


@dsl.pipeline(name="rfdetr-pipeline", pipeline_root="gs://my-project-vertex/pipeline")
def pipeline(epochs: int = 50) -> None:
    prep = preprocess_op()
    train = train_op(data=prep.outputs["data"], epochs=epochs)
    evaluate_op(model=train.outputs["model"])


aiplatform.PipelineJob(
    display_name="rfdetr-pipeline",
    template_path="pipeline.yaml",  # compiled from the @dsl.pipeline
    parameter_values={"epochs": 50},
).run()
```

## Conventions

- **Same container everywhere** — the image that trains locally is the one Vertex runs;
  never fork a Vertex-only training path.
- **Read `AIP_MODEL_DIR`/`AIP_*` env vars** for output paths instead of hardcoding GCS.
- **Pass secrets (W&B) via `environment_variables`**, never bake them into the image.
- **Right-size the accelerator** (`a2-highgpu-1g` + A100 for real training; `n1` + T4 for
  smoke tests) and set `boot_disk_size_gb` for large datasets.
- **Stage many-small-file datasets to local SSD** before training; stream large shards.

## Anti-patterns

- Vertex-specific training code that diverges from the local path — kills reproducibility.
- Hardcoding output paths instead of `AIP_MODEL_DIR` — artifacts lost when the VM dies.
- Baking API keys into the training image.
- Reading a 100k-image detection set file-by-file from `gs://` — IO-bound; copy to SSD.

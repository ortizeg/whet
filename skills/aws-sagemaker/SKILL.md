---
name: aws-sagemaker
description: >
  Use this skill when training or deploying ML models on AWS SageMaker — launching
  training jobs, standing up real-time endpoints or batch transform, running processing
  jobs, registering models, tuning hyperparameters, building SageMaker Pipelines, or
  wiring S3 data in and out. Reach for it any time the target is AWS-managed ML
  infrastructure, even if the user just says "train this in the cloud" and means AWS.
  Not for Google Cloud / Vertex AI training (see gcp) or self-managed Kubernetes
  (see kubernetes).
---

# AWS SageMaker Skill

Managed training, tuning, and deployment on AWS SageMaker. Use the SageMaker Python SDK for all operations — fall back to boto3 only when the SDK lacks a feature.

## Project Structure

```
project/
├── src/
│   ├── training/{train.py, model.py, data.py}   # training entry point + model/data
│   ├── inference/{inference.py, requirements.txt}  # model_fn/input_fn/predict_fn/output_fn
│   └── processing/preprocess.py                  # processing job script
├── pipelines/{training_pipeline.py, config.py}   # SageMaker Pipeline definitions
├── configs/{training.yaml, infrastructure.yaml}  # hyperparameters, instance types
└── tests/{test_training_local.py, test_inference.py}
```

## Essential Core: Launching a Training Job

Configure a PyTorch estimator with frozen Pydantic configs. Distribution is enabled automatically for multi-instance jobs. Submit with `wait=False` and poll — training jobs run for hours.

```python
"""SageMaker training job configuration."""

from __future__ import annotations

from pydantic import BaseModel
from sagemaker.pytorch import PyTorch


class TrainingConfig(BaseModel, frozen=True):
    role: str
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    max_run_seconds: int = 86400
    volume_size_gb: int = 100
    output_s3_uri: str
    base_job_name: str = "cv-training"
    framework_version: str = "2.1.0"
    py_version: str = "py310"


class HyperParameters(BaseModel, frozen=True):
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    model_name: str = "resnet50"
    num_classes: int = 10


def create_estimator(config: TrainingConfig, hp: HyperParameters) -> PyTorch:
    return PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role=config.role,
        instance_type=config.instance_type,
        instance_count=config.instance_count,
        framework_version=config.framework_version,
        py_version=config.py_version,
        output_path=config.output_s3_uri,
        base_job_name=config.base_job_name,
        max_run=config.max_run_seconds,
        volume_size=config.volume_size_gb,
        hyperparameters=hp.model_dump(),
        environment={"NCCL_DEBUG": "INFO", "TORCH_DISTRIBUTED_DEBUG": "DETAIL"},
        distribution={"torch_distributed": {"enabled": True}}
        if config.instance_count > 1
        else None,
    )


def launch_training(
    config: TrainingConfig, hp: HyperParameters, train_uri: str, val_uri: str
) -> str:
    """Launch async and return the job name (wait=False for long jobs)."""
    estimator = create_estimator(config, hp)
    estimator.fit(inputs={"train": train_uri, "validation": val_uri}, wait=False)
    return estimator.latest_training_job.name
```

## Conventions

- **Frozen Pydantic configs** for every job/endpoint/pipeline parameter set — no loose kwargs.
- **The training script reads `SM_*` environment variables as argparse defaults**, so the same
  script runs locally and in the cloud.
- **Keep `framework_version` / `py_version` identical** between training and inference so
  checkpoints load in the serving container.
- **One `inference.py`** (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) shared by real-time
  endpoints and batch transform.
- **Local mode first**: `instance_type="local"` with `file://` inputs, in CI, before any cloud job.
- **Pipelines take `ParameterString`/`ParameterFloat`** inputs and pass data between steps via
  `.properties` references, never literal S3 paths.
- **Register models as `PendingManualApproval`** so promotion stays a deliberate act.
- **Delete idle endpoints** — they bill continuously; use batch transform for offline scoring.

## Anti-Patterns

- **Never hardcode S3 paths** — use session defaults and `ParameterString` in pipelines.
- **Never oversize instances** — start at `ml.g5.xlarge` and scale up; don't reach for `ml.p3.16xlarge` for simple models.
- **Never skip local mode** — validate with `instance_type="local"` first.
- **Never put credentials in training scripts** — SageMaker injects the IAM role.
- **Never download the full dataset inside the script** — use `SM_CHANNEL_*` input channels.
- **Never block on long jobs** — submit with `wait=False` and poll status.
- **Never let training and inference preprocessing drift apart** — mismatched normalization is the most common cause of a "worse in production" model.

## Integration with Other Skills

- **PyTorch Lightning** — LightningModule inside training jobs; **Hydra Config** — hyperparameters flattened to key-value pairs.
- **W&B / MLflow** — experiment tracking in containers; **Docker CV** — custom containers when built-in images fall short.

## Deep dives

- `references/training-jobs.md` — read when writing `train.py`: `SM_*` environment variables, argparse wiring, distributed setup, instance sizing.
- `references/endpoints-and-inference.md` — read when deploying a real-time endpoint or writing the `model_fn`/`input_fn`/`predict_fn`/`output_fn` handlers.
- `references/batch-transform.md` — read when scoring an S3 dataset offline instead of standing up an endpoint.
- `references/hyperparameter-tuning.md` — read when running a Bayesian search, including the log-regex `metric_definitions` that make it work.
- `references/pipelines-and-model-registry.md` — read when composing preprocess → train → conditional-register into a SageMaker Pipeline, or versioning models in the registry.
- `references/s3-data-and-local-mode.md` — read when uploading datasets, pulling model artifacts back, or testing the training script with `instance_type="local"`.

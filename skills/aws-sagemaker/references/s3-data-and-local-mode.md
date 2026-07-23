# S3 Data Management and Local Mode Testing

Moving datasets and model artifacts in and out of S3, and validating a training script locally before paying for cloud instances.

## Contents

- [S3 Data Management](#s3-data-management)
- [Local Mode Testing](#local-mode-testing)

## S3 Data Management

Use the SageMaker session for uploads (auto-resolves the default bucket); use boto3 to pull artifacts back.

```python
from pathlib import Path

import boto3
import sagemaker
from loguru import logger


def upload_dataset(local_path: Path, bucket: str, prefix: str = "datasets") -> str:
    s3_uri = sagemaker.Session().upload_data(
        path=str(local_path), bucket=bucket, key_prefix=prefix
    )
    logger.info("Uploaded {} to {}", local_path, s3_uri)
    return s3_uri


def download_model_artifacts(model_data_s3: str, local_dir: Path) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    bucket, key = model_data_s3.replace("s3://", "").split("/", 1)
    local_path = local_dir / Path(key).name
    boto3.resource("s3").Bucket(bucket).download_file(key, str(local_path))
    logger.info("Downloaded {} to {}", model_data_s3, local_path)
    return local_path
```

Notes:

- `Session().upload_data` returns the full `s3://` URI, which is exactly what
  `estimator.fit(inputs={"train": uri})` expects. Omit `bucket` to use the account's default
  SageMaker bucket rather than hardcoding one.
- Training artifacts arrive as `model.tar.gz`; `download_model_artifacts` fetches the object,
  then untar it locally to inspect weights or metrics.
- Keep a stable prefix convention (`datasets/<name>/<version>/`) so pipeline parameters can
  point at a dataset version rather than an ad-hoc path.
- The training job never reaches into S3 itself — it reads from `SM_CHANNEL_*` paths that
  SageMaker populated from these URIs. Downloading inside the script bypasses the channel
  mechanism and breaks local mode.
- For very large datasets, prefer `input_mode="FastFile"` or Pipe mode on the estimator so the
  job streams from S3 instead of waiting for a full EBS download.

## Local Mode Testing

Test the training script with `instance_type="local"` and `file://` inputs before submitting cloud jobs:

```python
import pytest
from sagemaker.pytorch import PyTorch


@pytest.fixture
def local_estimator() -> PyTorch:
    return PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role="arn:aws:iam::000000000000:role/dummy",
        instance_type="local",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
        hyperparameters={"epochs": 1, "batch-size": 4, "model-name": "resnet18"},
    )


def test_training_local(local_estimator: PyTorch, tmp_path: Path) -> None:
    create_test_dataset(tmp_path / "train")
    create_test_dataset(tmp_path / "val")
    local_estimator.fit({
        "train": f"file://{tmp_path / 'train'}",
        "validation": f"file://{tmp_path / 'val'}",
    })
```

Notes:

- Local mode runs the **real DLC container** on the local Docker daemon, so it catches the
  failures that matter: missing dependencies, wrong entry-point path, `SM_*` misuse, and
  serialization bugs. Docker must be installed and running.
- The IAM role is a dummy ARN — local mode never calls the SageMaker control plane.
- Use a tiny model and one epoch (`resnet18`, `epochs=1`, `batch-size=4`) with a synthetic
  dataset in `tmp_path`. The point is to prove the script runs end to end, not to train anything.
- `instance_type="local_gpu"` exercises the GPU path if the dev machine has one and the NVIDIA
  container runtime is installed.
- Run this in CI before any cloud submission — a typo caught here costs seconds instead of a
  failed multi-hour GPU job.

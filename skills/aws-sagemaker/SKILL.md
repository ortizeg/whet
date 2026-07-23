---
name: aws-sagemaker
description: >
  Use this skill when training or deploying ML models on AWS SageMaker — launching
  training jobs, standing up real-time endpoints or batch transform, running processing
  jobs, registering models, tuning hyperparameters, building SageMaker Pipelines, or
  wiring S3 data in and out. Reach for it any time the target is AWS-managed ML
  infrastructure, even if the user just says "train this in the cloud" and means AWS.
  Not for Google Cloud / Vertex AI training (see vertex-ai) or self-managed Kubernetes
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

## Training Jobs

Configure a PyTorch estimator with frozen Pydantic configs. Distribution is enabled automatically for multi-instance jobs.

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

### Training Script Entry Point

SageMaker injects channels and paths via `SM_*` environment variables. Read them as argparse defaults, never hardcode paths.

```python
"""src/training/train.py"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--model-name", type=str, default="resnet50")
    # SageMaker-injected paths
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    logger.info("Starting training: {}", vars(args))
    world_size = int(os.environ.get("SM_NUM_GPUS", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, args.num_classes).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ... training loop ...

    torch.save(model.state_dict(), Path(args.model_dir) / "model.pth")
    metrics = {"final_val_loss": 0.25, "final_val_acc": 0.92}
    (Path(args.output_data_dir) / "metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    train(parse_args())
```

## Real-Time Endpoints

Deploy a `PyTorchModel` with a custom inference script:

```python
from pydantic import BaseModel
from sagemaker.pytorch import PyTorchModel


class EndpointConfig(BaseModel, frozen=True):
    role: str
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    endpoint_name: str
    model_data_s3: str
    framework_version: str = "2.1.0"
    py_version: str = "py310"


def deploy_endpoint(config: EndpointConfig) -> str:
    model = PyTorchModel(
        model_data=config.model_data_s3,
        role=config.role,
        framework_version=config.framework_version,
        py_version=config.py_version,
        entry_point="inference.py",
        source_dir="src/inference",
    )
    predictor = model.deploy(
        initial_instance_count=config.instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
    )
    return predictor.endpoint_name
```

### Custom Inference Handlers

SageMaker calls these in order: `model_fn` (once at load), then `input_fn → predict_fn → output_fn` per request.

```python
"""src/inference/inference.py"""

from __future__ import annotations

import io
import json

import torch
from PIL import Image
from torchvision import transforms


def model_fn(model_dir: str) -> torch.nn.Module:
    model = build_model("resnet50", num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tfm(image).unsqueeze(0)
    if content_type == "application/json":
        return torch.tensor(json.loads(request_body)["instances"])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> dict:
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(input_data.to(device))
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    return {
        "predictions": preds.cpu().numpy().tolist(),
        "probabilities": probs.cpu().numpy().tolist(),
    }


def output_fn(prediction: dict, accept: str) -> str:
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
```

## Hyperparameter Tuning

Bayesian search over parameter ranges. Metrics are parsed from training logs via regex `metric_definitions`.

```python
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)


def create_tuner(estimator: PyTorch) -> HyperparameterTuner:
    return HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:accuracy",
        objective_type="Maximize",
        hyperparameter_ranges={
            "learning-rate": ContinuousParameter(1e-5, 1e-2, scaling_type="Logarithmic"),
            "batch-size": CategoricalParameter([16, 32, 64, 128]),
            "epochs": IntegerParameter(10, 100),
        },
        metric_definitions=[
            {"Name": "validation:accuracy", "Regex": r"val_acc=(\S+)"},
            {"Name": "validation:loss", "Regex": r"val_loss=(\S+)"},
        ],
        max_jobs=20,
        max_parallel_jobs=4,
        strategy="Bayesian",
    )
```

## SageMaker Pipelines

End-to-end preprocess → train → conditional-register. Steps pass data via `.properties` references; registration only runs if accuracy clears a threshold.

```python
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep


def create_pipeline(role: str, pipeline_name: str = "cv-training-pipeline") -> Pipeline:
    session = sagemaker.Session()
    input_data = ParameterString(name="InputData")
    accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.90)

    processor = ScriptProcessor(
        role=role,
        image_uri=session.sagemaker_client.describe_image("pytorch-training")["ImageUri"],
        instance_type="ml.m5.xlarge",
        instance_count=1,
        command=["python3"],
    )
    preprocess = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="val", source="/opt/ml/processing/output/val"),
        ],
        code="src/processing/preprocess.py",
    )

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src/training",
        role=role,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        framework_version="2.1.0",
        py_version="py310",
    )
    outs = preprocess.properties.ProcessingOutputConfig.Outputs
    train_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={"train": outs["train"].S3Output.S3Uri, "validation": outs["val"].S3Output.S3Uri},
    )
    register = ModelStep(
        name="RegisterModel",
        step_args=estimator.register(
            content_types=["application/json"],
            response_types=["application/json"],
            model_package_group_name="cv-models",
            approval_status="PendingManualApproval",
        ),
    )
    condition = ConditionStep(
        name="CheckAccuracy",
        conditions=[ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=train_step.name, property_file="metrics", json_path="val_accuracy"),
            right=accuracy_threshold,
        )],
        if_steps=[register],
        else_steps=[],
    )
    return Pipeline(
        name=pipeline_name,
        parameters=[input_data, accuracy_threshold],
        steps=[preprocess, train_step, condition],
        sagemaker_session=session,
    )
```

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

## Batch Transform

Offline inference over an S3 dataset — reuses the same `PyTorchModel`/inference script as endpoints:

```python
from sagemaker.pytorch import PyTorchModel


def run_batch_transform(
    model_data_s3: str, input_s3_uri: str, output_s3_uri: str, role: str,
    instance_type: str = "ml.g5.xlarge",
) -> None:
    model = PyTorchModel(
        model_data=model_data_s3, role=role,
        framework_version="2.1.0", py_version="py310",
        entry_point="inference.py", source_dir="src/inference",
    )
    transformer = model.transformer(
        instance_count=1, instance_type=instance_type,
        output_path=output_s3_uri, strategy="MultiRecord", max_payload=6,
    )
    transformer.transform(data=input_s3_uri, content_type="application/json", split_type="Line")
```

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

## Anti-Patterns

- **Never hardcode S3 paths** — use session defaults and `ParameterString` in pipelines.
- **Never oversize instances** — start at `ml.g5.xlarge` and scale up; don't reach for `ml.p3.16xlarge` for simple models.
- **Never skip local mode** — validate with `instance_type="local"` first.
- **Never put credentials in training scripts** — SageMaker injects the IAM role.
- **Never download the full dataset inside the script** — use `SM_CHANNEL_*` input channels.
- **Never block on long jobs** — submit with `wait=False` and poll status.

## Integration with Other Skills

- **PyTorch Lightning** — LightningModule inside training jobs; **Hydra Config** — hyperparameters flattened to key-value pairs.
- **W&B / MLflow** — experiment tracking in containers; **Docker CV** — custom containers when built-in images fall short.
- **DVC** — data versioning with an S3 remote SageMaker can access.

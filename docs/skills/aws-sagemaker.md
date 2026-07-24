# AWS SageMaker

The AWS SageMaker skill provides expert patterns for ML training and deployment on Amazon SageMaker, covering training jobs, endpoints, pipelines, and hyperparameter tuning.

**Skill directory:** `skills/aws-sagemaker/`

## Purpose

SageMaker is the standard managed ML platform on AWS. This skill encodes best practices for structuring SageMaker projects: PyTorch estimator configuration with distributed training, custom inference handlers, SageMaker Pipelines with conditional model registration, hyperparameter tuning jobs, and local mode testing.

## When to Use

Use this skill whenever you need to:

- Train models on AWS GPU instances (ml.g5, ml.p4, ml.trn1)
- Deploy models as real-time SageMaker endpoints with auto-scaling
- Build automated ML pipelines (preprocessing, training, evaluation, registration)
- Run hyperparameter tuning to optimize model performance
- Process large datasets with SageMaker Processing jobs

## Key Patterns

### Training Job with PyTorch Estimator

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src/training",
    role=config.role,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    framework_version="2.1.0",
    py_version="py310",
    output_path=config.output_s3_uri,
    hyperparameters=hyperparameters.model_dump(),
)

estimator.fit(
    inputs={"train": train_s3_uri, "validation": val_s3_uri},
    wait=False,
)
```

### Custom Inference Handler

```python
def model_fn(model_dir: str) -> torch.nn.Module:
    """Load trained model."""
    model = build_model("resnet50", num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth"))
    model.eval()
    return model.cuda()


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    """Deserialize input to tensor."""
    ...


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> dict:
    """Run inference."""
    ...
```

### SageMaker Pipeline

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep

pipeline = Pipeline(
    name="cv-training-pipeline",
    parameters=[input_data, accuracy_threshold],
    steps=[preprocess_step, training_step, condition_step],
)
```

## Anti-Patterns to Avoid

- Do not hardcode S3 paths -- use SageMaker session defaults and pipeline parameters
- Do not skip local mode testing -- validate with `instance_type="local"` first
- Do not put credentials in training scripts -- SageMaker injects IAM roles
- Do not download datasets inside training scripts -- use SageMaker input channels

## Combines Well With

- **PyTorch Lightning** -- LightningModule inside SageMaker training jobs
- **Docker CV** -- Custom training containers when default images are insufficient
- **W&B / MLflow** -- Experiment tracking inside SageMaker training containers
- **GCP** -- Alternative cloud platform patterns for comparison

## Full Reference

See [`skills/aws-sagemaker/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/aws-sagemaker/SKILL.md) for complete patterns including batch transform, hyperparameter tuning, and S3 data management.

# Model Registry

Scope: registering models from a run, moving versions through stages, annotating versions, and loading a registered model back.

The model registry provides a centralized model store with versioning and stage transitions.

## Registering a model

```python
import mlflow
import mlflow.pytorch

# Log and register model in one step
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log PyTorch model and register it
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="yolov8-coco",
    )
```

## Managing model versions

```python
from mlflow import MlflowClient

client = MlflowClient()

# Transition model to staging
client.transition_model_version_stage(
    name="yolov8-coco",
    version=3,
    stage="Staging",
)

# Promote to production
client.transition_model_version_stage(
    name="yolov8-coco",
    version=3,
    stage="Production",
)

# Add description
client.update_model_version(
    name="yolov8-coco",
    version=3,
    description="Best model with mAP=0.52 on COCO val2017",
)
```

## Loading a registered model

```python
import mlflow.pytorch

# Load by stage
model = mlflow.pytorch.load_model("models:/yolov8-coco/Production")

# Load by version
model = mlflow.pytorch.load_model("models:/yolov8-coco/3")

# Load by run ID
model = mlflow.pytorch.load_model("runs:/abc123def456/model")
```

Prefer stage URIs (`models:/name/Production`) in deployment code so promoting a new
version does not require a code change, and pin an explicit version URI when you need
a reproducible evaluation of one specific model.

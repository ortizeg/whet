# W&B Artifacts for Models and Datasets

Scope: versioning model checkpoints, datasets, and prediction files as W&B artifacts, and pulling them back down.

Artifacts let you version datasets, models, and other files alongside your runs.

## Saving a Model Artifact

```python
import wandb

def save_model_artifact(
    model_path: str,
    artifact_name: str,
    metadata: dict,
) -> None:
    """Save model as a W&B artifact."""
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata,
        description=f"Model checkpoint with mAP={metadata.get('mAP', 'N/A')}",
    )
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)

# Usage
save_model_artifact(
    model_path="checkpoints/best_model.pt",
    artifact_name="yolov8-coco",
    metadata={"mAP": 0.45, "epoch": 50, "image_size": 640},
)
```

## Loading an Artifact

```python
import wandb

def load_model_artifact(artifact_path: str, download_dir: str) -> str:
    """Download a model artifact and return the local path."""
    run = wandb.init(project="my-cv-project")
    artifact = run.use_artifact(artifact_path)
    local_dir = artifact.download(root=download_dir)
    return local_dir

# Usage
model_dir = load_model_artifact(
    artifact_path="my-team/my-cv-project/yolov8-coco:v3",
    download_dir="./downloaded_models",
)
```

## Dataset Artifacts

```python
import wandb

def create_dataset_artifact(
    data_dir: str,
    name: str,
    split: str,
) -> None:
    """Create a dataset artifact from a directory."""
    artifact = wandb.Artifact(
        name=f"{name}-{split}",
        type="dataset",
        metadata={"split": split},
    )
    artifact.add_dir(data_dir)
    wandb.log_artifact(artifact)
```

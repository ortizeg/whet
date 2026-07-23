# W&B Model Registry

Scope: promoting logged model artifacts into the W&B Model Registry and aliasing versions for downstream consumers.

The W&B Model Registry provides a central place to manage model versions and promote them through stages:

```python
import wandb

# Link a model artifact to the registry
run = wandb.init(project="my-cv-project")
run.link_artifact(
    artifact=model_artifact,
    target_path="my-team/wandb-registry-model/yolov8-production",
    aliases=["best", "v1.2"],
)
```

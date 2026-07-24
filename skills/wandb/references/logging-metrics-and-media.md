# Logging Metrics, Images, Video, and Alerts to W&B

Scope: everything that goes through `wandb.log` — scalar metrics, learning rate and gradients, custom step axes, bounding-box and segmentation overlays, video, and run alerts.

## Contents

- [Rich Run Initialization](#rich-run-initialization)
- [Logging Scalars](#logging-scalars)
- [Logging Learning Rate and Gradients](#logging-learning-rate-and-gradients)
- [Custom Step Tracking](#custom-step-tracking)
- [Logging Predictions with Bounding Boxes](#logging-predictions-with-bounding-boxes)
- [Logging Segmentation Masks](#logging-segmentation-masks)
- [Logging Video](#logging-video)
- [Alerts](#alerts)

## Rich Run Initialization

```python
import wandb

# Initialize a new run
run = wandb.init(
    project="my-cv-project",
    name="yolov8-experiment-001",
    config={
        "model": "yolov8",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealing",
        "image_size": 640,
    },
    tags=["baseline", "yolov8", "coco"],
    notes="Baseline YOLOv8 training on COCO subset",
)
```

## Logging Scalars

```python
import wandb

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_map = evaluate(model, val_loader)

    # Log metrics per epoch
    wandb.log({
        "train/loss": train_loss,
        "val/loss": val_loss,
        "val/mAP": val_map,
        "epoch": epoch,
    })
```

## Logging Learning Rate and Gradients

```python
# Log learning rate from scheduler
wandb.log({
    "lr": optimizer.param_groups[0]["lr"],
    "grad_norm": compute_grad_norm(model),
})

# Watch model for automatic gradient and parameter logging
wandb.watch(model, log="all", log_freq=100)
```

## Custom Step Tracking

```python
# Use custom x-axis
wandb.define_metric("train/loss", step_metric="global_step")
wandb.define_metric("val/*", step_metric="epoch")

for step, batch in enumerate(train_loader):
    loss = train_step(model, batch)
    wandb.log({"train/loss": loss, "global_step": step})
```

## Logging Predictions with Bounding Boxes

```python
import wandb
import numpy as np

def log_predictions(
    images: list[np.ndarray],
    predictions: list[dict],
    class_names: list[str],
    max_images: int = 16,
) -> None:
    """Log prediction images with bounding boxes to W&B."""
    logged_images = []

    for img, pred in zip(images[:max_images], predictions[:max_images]):
        box_data = []
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            box_data.append({
                "position": {
                    "minX": float(box[0]),
                    "minY": float(box[1]),
                    "maxX": float(box[2]),
                    "maxY": float(box[3]),
                },
                "class_id": int(label),
                "box_caption": f"{class_names[label]}: {score:.2f}",
                "scores": {"confidence": float(score)},
            })

        logged_images.append(
            wandb.Image(
                img,
                boxes={"predictions": {
                    "box_data": box_data,
                    "class_labels": {i: name for i, name in enumerate(class_names)},
                }},
            )
        )

    wandb.log({"predictions": logged_images})
```

## Logging Segmentation Masks

```python
import wandb
import numpy as np

def log_segmentation(
    image: np.ndarray,
    mask_pred: np.ndarray,
    mask_gt: np.ndarray,
    class_labels: dict[int, str],
) -> None:
    """Log segmentation predictions and ground truth."""
    wandb.log({
        "segmentation": wandb.Image(
            image,
            masks={
                "predictions": {"mask_data": mask_pred, "class_labels": class_labels},
                "ground_truth": {"mask_data": mask_gt, "class_labels": class_labels},
            },
        )
    })
```

## Logging Video

```python
import wandb
import numpy as np

# Log video as a sequence of frames (T, C, H, W)
frames = np.random.randint(0, 255, (30, 3, 480, 640), dtype=np.uint8)
wandb.log({"video": wandb.Video(frames, fps=10, format="mp4")})
```

## Alerts

Use `wandb.alert()` to get notified when training finishes or metrics degrade.

```python
# Alert on training completion
wandb.alert(
    title="Training Complete",
    text=f"Final mAP: {final_map:.4f}",
    level=wandb.AlertLevel.INFO,
)
```

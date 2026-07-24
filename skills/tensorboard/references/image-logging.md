# Image Logging

Scope: writing single images, image grids, and matplotlib figures (e.g. detection overlays) to TensorBoard.

TensorBoard can display images, which is invaluable for computer vision projects.

## Logging Individual Images

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log a single image (C, H, W) format, values in [0, 1]
writer.add_image("sample/input", image_tensor, epoch)

# Log prediction vs ground truth
writer.add_image("sample/prediction", pred_image, epoch)
writer.add_image("sample/ground_truth", gt_image, epoch)
```

## Logging Image Grids

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter(log_dir="logs/experiment_001")

# Create a grid of images (N, C, H, W)
images = torch.stack([batch[i] for i in range(min(16, len(batch)))])
grid = make_grid(images, nrow=4, normalize=True, padding=2)
writer.add_image("batch/inputs", grid, epoch)

# Log augmented vs original
original_grid = make_grid(original_images[:8], nrow=4, normalize=True)
augmented_grid = make_grid(augmented_images[:8], nrow=4, normalize=True)
writer.add_image("augmentation/original", original_grid, epoch)
writer.add_image("augmentation/augmented", augmented_grid, epoch)
```

## Logging Images with Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

def log_detection_figure(
    writer: SummaryWriter,
    image: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    step: int,
) -> None:
    """Log detection results as a matplotlib figure."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color="red", fontsize=10)
    ax.axis("off")
    writer.add_figure("detections", fig, step)
    plt.close(fig)
```

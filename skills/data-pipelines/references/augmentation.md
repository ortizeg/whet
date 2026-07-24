# Augmentation Pipeline Design

Config-driven Albumentations pipelines, with a single strength knob and strictly deterministic val/test transforms.

## Principle

Augmentation is part of the data contract, not a training detail: configure it,
scale it with a single `strength` knob, and keep val/test strictly deterministic.

## Config-Driven Albumentations

```python
"""Albumentations pipelines driven by config."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel, Field

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class AugmentationConfig(BaseModel):
    image_size: int = Field(ge=32, default=640)
    strength: float = Field(ge=0.0, le=1.0, default=0.5)


def build_train_transforms(cfg: AugmentationConfig) -> A.Compose:
    """Training augmentations, all probabilities scaled by a single strength knob."""
    s, size = cfg.strength, cfg.image_size
    return A.Compose(
        [
            A.RandomResizedCrop(height=size, width=size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2 * s, contrast=0.2 * s, saturation=0.2 * s, p=0.8),
            A.GaussNoise(p=0.3 * s),
            A.CoarseDropout(max_holes=int(8 * s), p=0.3 * s),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )
```

## Val/Test Transforms

Val/test transforms are the same `Compose` with everything random removed —
`A.Resize`, `A.Normalize`, `ToTensorV2`, nothing else. Log the augmentation config
with the run: an unrecorded `strength` change is an unreproducible run.

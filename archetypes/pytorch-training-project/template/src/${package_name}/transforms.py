"""Image transform pipelines for ${project_name}."""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size: int) -> transforms.Compose:
    """Build the augmentation pipeline used for the training split.

    Args:
        image_size: Target square resolution in pixels.

    Returns:
        A composed transform producing normalized ``float32`` tensors.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms(image_size: int) -> transforms.Compose:
    """Build the deterministic pipeline used for validation and testing.

    Args:
        image_size: Target square resolution in pixels.

    Returns:
        A composed transform producing normalized ``float32`` tensors.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

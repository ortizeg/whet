# LightningDataModule Design

The full reference LightningDataModule for image classification, including transforms, `setup`, dataloaders, and the rules that govern them.

## Contents

- [LightningDataModule Patterns](#lightningdatamodule-patterns)
- [Key Rules for DataModule](#key-rules-for-datamodule)

## LightningDataModule Patterns

Wrap every dataset in a `LightningDataModule` to separate data logic from model logic and ensure reproducibility.

```python
"""DataModule for image classification datasets."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageClassificationDataModule(L.LightningDataModule):
    """DataModule for image classification."""

    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: tuple[int, int] = (224, 224),
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    @property
    def train_transform(self) -> transforms.Compose:
        """Training augmentations."""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def val_transform(self) -> transforms.Compose:
        """Validation/test transforms (no augmentation)."""
        return transforms.Compose([
            transforms.Resize(self.image_size[0] + 32),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolder(
                self.data_dir / "train",
                transform=self.train_transform,
            )
            self.val_dataset = ImageFolder(
                self.data_dir / "val",
                transform=self.val_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageFolder(
                self.data_dir / "test",
                transform=self.val_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    # test_dataloader mirrors val_dataloader with self.test_dataset (shuffle=False).
```

## Key Rules for DataModule

1. **Always use `setup(stage)`** to initialize datasets -- never in `__init__`.
2. **Use `persistent_workers=True`** when `num_workers > 0` to avoid re-forking workers each epoch.
3. **Use `drop_last=True`** for training to avoid batch normalization issues with tiny final batches.
4. **Separate train and val transforms** -- validation must never include random augmentations.
5. **Use `pin_memory=True`** for GPU training to speed up host-to-device transfers.
6. **Put data downloads in `prepare_data()`**, not `setup()` -- `prepare_data()` runs on rank 0 only.

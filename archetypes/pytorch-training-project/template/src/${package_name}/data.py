"""LightningDataModule for ${project_name}."""

from __future__ import annotations

import lightning as L
from loguru import logger
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .transforms import eval_transforms, train_transforms

# Batches are ``(images, labels)`` tuples — the shape produced by torchvision
# datasets under PyTorch's default collate function.
Batch = tuple[Tensor, Tensor]


class DataConfig(BaseModel, frozen=True):
    """Data configuration."""

    data_dir: str = "data"
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    num_classes: int = Field(default=10, ge=2)
    image_size: int = Field(default=224, ge=8)
    train_size: int = Field(default=100, ge=1)
    val_size: int = Field(default=20, ge=1)


class ImageDataModule(L.LightningDataModule):
    """Image dataset module.

    Args:
        config: Data configuration.
    """

    def __init__(self, config: DataConfig) -> None:
        super().__init__()
        self.save_hyperparameters(config.model_dump())
        self.config = config
        self.train_dataset: Dataset[Batch] | None = None
        self.val_dataset: Dataset[Batch] | None = None

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for each stage."""
        # Replace ``FakeData`` with your own dataset. Whatever you use, keep the
        # ``(image, label)`` tuple contract that the LightningModule unpacks.
        from torchvision import datasets

        if stage in {"fit", "validate", None}:
            image_shape = (3, self.config.image_size, self.config.image_size)
            self.train_dataset = datasets.FakeData(
                size=self.config.train_size,
                image_size=image_shape,
                num_classes=self.config.num_classes,
                transform=train_transforms(self.config.image_size),
            )
            self.val_dataset = datasets.FakeData(
                size=self.config.val_size,
                image_size=image_shape,
                num_classes=self.config.num_classes,
                transform=eval_transforms(self.config.image_size),
            )
            logger.info(
                "Datasets ready | train={} val={}",
                self.config.train_size,
                self.config.val_size,
            )

    def train_dataloader(self) -> DataLoader[Batch]:
        """Return the training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("setup('fit') must run before train_dataloader()")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        """Return the validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError("setup('fit') must run before val_dataloader()")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

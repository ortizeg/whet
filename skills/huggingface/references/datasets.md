# Hugging Face Datasets

Scope: loading, preprocessing, and tokenizing datasets with the `datasets` library, including local image folders.

## Loading and Processing Datasets

```python
"""Dataset loading and preprocessing with Hugging Face datasets."""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoImageProcessor, AutoTokenizer
from loguru import logger


def load_image_dataset(
    dataset_name: str,
    processor: AutoImageProcessor,
    split: str | None = None,
) -> DatasetDict | Dataset:
    """Load and preprocess an image classification dataset."""
    dataset = load_dataset(dataset_name, split=split)
    logger.info("Loaded dataset: {} rows", len(dataset) if isinstance(dataset, Dataset) else sum(len(s) for s in dataset.values()))

    def preprocess(batch: dict) -> dict:
        images = batch["image"]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    processed = dataset.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=dataset.column_names if isinstance(dataset, Dataset) else dataset["train"].column_names,
    )

    processed.set_format("torch")
    return processed


def load_text_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> DatasetDict:
    """Load and tokenize a text dataset."""
    dataset = load_dataset(dataset_name)

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    return tokenized


def create_custom_dataset(
    data_dir: str,
    image_column: str = "image",
    label_column: str = "label",
) -> Dataset:
    """Create a dataset from a local directory of images."""
    dataset = load_dataset(
        "imagefolder",
        data_dir=data_dir,
    )
    logger.info("Created dataset from {}: {} images", data_dir, len(dataset["train"]))
    return dataset["train"]
```

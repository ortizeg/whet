---
name: huggingface
description: >
  Use this skill when working with the Hugging Face ecosystem for NLP or vision — loading
  Transformers models with AutoModel, the datasets library, tokenizers, task pipelines,
  fine-tuning with Trainer, PEFT/LoRA adapters, publishing to the Model Hub, and
  quantization/inference optimization. Reach for it any time a pretrained transformer, an
  HF dataset, or the Hub is involved, even if the user doesn't say "Hugging Face". For the
  training-loop framework around these models see pytorch-lightning.
---

# Hugging Face Skill

Patterns for the Hugging Face ecosystem (Transformers, Datasets, Tokenizers, PEFT). Use `Auto*` classes for loading (never hardcode model class names), the `datasets` library for data, and the `Trainer` API for fine-tuning unless you need a custom loop.

## Model Loading

### AutoModel Pattern

```python
"""Loading pretrained models with Hugging Face Transformers."""

from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    AutoModelForObjectDetection,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoImageProcessor,
)
from pydantic import BaseModel, Field
from loguru import logger


class ModelConfig(BaseModel, frozen=True):
    """Hugging Face model configuration."""

    model_name: str = "microsoft/resnet-50"
    revision: str = "main"
    torch_dtype: str = "float32"
    device_map: str | None = None
    trust_remote_code: bool = False
    cache_dir: str | None = None


def load_vision_model(config: ModelConfig) -> tuple:
    """Load a vision model and its image processor."""
    logger.info("Loading model: {}", config.model_name)

    processor = AutoImageProcessor.from_pretrained(
        config.model_name,
        revision=config.revision,
        cache_dir=config.cache_dir,
    )

    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        revision=config.revision,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
        cache_dir=config.cache_dir,
    )

    logger.info("Model loaded: {} parameters", sum(p.numel() for p in model.parameters()))
    return model, processor


def load_text_model(config: ModelConfig) -> tuple:
    """Load a text model and tokenizer (same shape as the vision path)."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.revision,
        torch_dtype=getattr(torch, config.torch_dtype),
        device_map=config.device_map,
        cache_dir=config.cache_dir,
    )
    return model, tokenizer
```

## Datasets Library

### Loading and Processing Datasets

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

## Fine-Tuning with Trainer

### Standard Trainer Configuration

```python
"""Fine-tuning with Hugging Face Trainer API."""

from __future__ import annotations

from pathlib import Path

import evaluate
import numpy as np
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from loguru import logger


class FinetuneConfig(BaseModel, frozen=True):
    """Fine-tuning configuration."""

    model_name: str = "microsoft/resnet-50"
    output_dir: str = "outputs/finetuned"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    push_to_hub: bool = False
    hub_model_id: str | None = None


def create_training_args(config: FinetuneConfig) -> TrainingArguments:
    """Map config fields directly onto TrainingArguments, plus fixed extras."""
    return TrainingArguments(
        **config.model_dump(),
        logging_steps=50,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute accuracy for evaluation."""
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predicted = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predicted, references=labels)


def finetune(
    config: FinetuneConfig,
    train_dataset,
    eval_dataset,
    num_labels: int,
) -> str:
    """Fine-tune a pretrained model."""
    logger.info("Fine-tuning {} for {} epochs", config.model_name, config.num_train_epochs)

    model = AutoModelForImageClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    training_args = create_training_args(config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save final model
    output_path = Path(config.output_dir) / "best"
    trainer.save_model(str(output_path))
    logger.info("Best model saved to {}", output_path)

    return str(output_path)
```

## PEFT / LoRA

### Parameter-Efficient Fine-Tuning

```python
"""PEFT fine-tuning with LoRA adapters."""

from __future__ import annotations

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)
from transformers import AutoModelForSequenceClassification
from loguru import logger


def create_lora_model(
    model_name: str,
    num_labels: int,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> PeftModel:
    """Create a LoRA-adapted model for fine-tuning."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or ["query", "value"],
        bias="none",
    )

    model = get_peft_model(base_model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA model: {:.1f}M trainable / {:.1f}M total ({:.1f}%)",
        trainable / 1e6,
        total / 1e6,
        100 * trainable / total,
    )

    return model


def load_lora_model(
    base_model_name: str,
    adapter_path: str,
    num_labels: int,
) -> PeftModel:
    """Load a saved LoRA adapter on top of a base model."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model
```

## Pipelines for Quick Inference

### Using Pipeline API

```python
"""Hugging Face pipeline patterns for quick inference."""

from __future__ import annotations

from transformers import pipeline


def create_image_classifier(
    model_name: str = "microsoft/resnet-50",
    device: int = 0,
) -> pipeline:
    """Create an image classification pipeline."""
    return pipeline("image-classification", model=model_name, device=device)


# Same pattern for other tasks — just change the task string and default model:
#   "object-detection"              (facebook/detr-resnet-50, add threshold=0.5)
#   "zero-shot-image-classification" (openai/clip-vit-base-patch32)
def create_object_detector(model_name: str = "facebook/detr-resnet-50", device: int = 0) -> pipeline:
    """Create an object detection pipeline."""
    return pipeline("object-detection", model=model_name, device=device, threshold=0.5)


# Usage: classifier(...) -> [{"label": "cat", "score": 0.97}, ...]
#        detector(...)   -> [{"label": "person", "score": 0.99, "box": {...}}, ...]
classifier = create_image_classifier()
results = classifier("path/to/image.jpg")
```

## Model Hub Publishing

### Pushing Models to the Hub

```python
"""Publishing models and datasets to Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, ModelCard, ModelCardData
from transformers import AutoModel, AutoTokenizer
from loguru import logger


def push_model_to_hub(
    model_path: str,
    repo_id: str,
    private: bool = True,
) -> str:
    """Push a trained model to Hugging Face Hub."""
    model = AutoModel.from_pretrained(model_path)
    model.push_to_hub(repo_id, private=private)

    # Push tokenizer/processor if present
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.push_to_hub(repo_id, private=private)
    except Exception:
        pass

    logger.info("Model pushed to hub: {}", repo_id)
    return f"https://huggingface.co/{repo_id}"


def create_model_card(
    repo_id: str, model_name: str, dataset_name: str, metrics: dict[str, float]
) -> ModelCard:
    """Create and push a model card (always include license, dataset, metrics)."""
    card_data = ModelCardData(
        language="en",
        license="apache-2.0",
        model_name=model_name,
        datasets=[dataset_name],
        metrics=list(metrics.keys()),
    )
    card = ModelCard.from_template(
        card_data,
        model_id=repo_id,
        model_description=f"Fine-tuned {model_name} on {dataset_name}.",
        training_metrics=metrics,
    )
    card.push_to_hub(repo_id)
    return card
```

## Quantization and Optimization

### Model Quantization for Faster Inference

```python
"""Model quantization with Hugging Face Optimum."""

from __future__ import annotations

import torch
from transformers import AutoModelForImageClassification, BitsAndBytesConfig
from loguru import logger


def load_quantized_model(
    model_name: str,
    num_labels: int,
    load_in_4bit: bool = True,
) -> AutoModelForImageClassification:
    """Load a 4-bit or 8-bit quantized model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info("Loaded quantized model ({} bit)", "4" if load_in_4bit else "8")
    return model
```

## Anti-Patterns

- **Never hardcode model class names** — use `AutoModel`, `AutoTokenizer`, `AutoImageProcessor` for flexibility.
- **Never download models inside training loops** — load once at initialization, cache with `cache_dir`.
- **Never skip `revision` parameter** — pin model versions for reproducibility.
- **Never tokenize the full dataset eagerly** — use `dataset.map(batched=True)` with lazy loading.
- **Never fine-tune all parameters when data is limited** — use PEFT/LoRA to reduce overfitting.
- **Never push models to public hub without a model card** — always include metrics, dataset, and license info.
- **Never ignore `trust_remote_code` warnings** — only set to `True` for trusted model sources.

## Integration with Other Skills

Wrap HF models in a Lightning Trainer for custom loops; set `report_to=["wandb"]` for tracking; export via `optimum`/ONNX for inference; deploy on SageMaker (HF DLC) or behind FastAPI; version Hub datasets with DVC.

# Fine-Tuning with the Trainer API

Scope: a typed `TrainingArguments` config, metric computation, and a complete `Trainer` fine-tuning run with early stopping.

## Standard Trainer Configuration

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

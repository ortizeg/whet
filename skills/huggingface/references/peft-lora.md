# PEFT / LoRA Adapters

Scope: parameter-efficient fine-tuning — wrapping a base model with a LoRA adapter and reloading a saved adapter.

## Parameter-Efficient Fine-Tuning

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

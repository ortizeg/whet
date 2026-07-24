# Loading Transformers Models

Scope: the `Auto*` loading pattern for vision and text models, with a typed Pydantic model config.

## AutoModel Pattern

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

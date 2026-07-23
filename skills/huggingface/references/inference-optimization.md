# Inference: Pipelines and Quantization

Scope: the `pipeline` API for quick task inference, and 4-bit/8-bit quantized model loading for faster, smaller inference.

## Contents

- [Using Pipeline API](#using-pipeline-api)
- [Model Quantization for Faster Inference](#model-quantization-for-faster-inference)

## Using Pipeline API

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

## Model Quantization for Faster Inference

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

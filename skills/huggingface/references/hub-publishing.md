# Publishing to the Hugging Face Hub

Scope: pushing trained models (and their tokenizer/processor) to the Hub and generating a model card.

## Pushing Models to the Hub

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

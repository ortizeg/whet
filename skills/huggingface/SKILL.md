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

Patterns for the Hugging Face ecosystem (Transformers, Datasets, Tokenizers, PEFT). Use
`Auto*` classes for loading (never hardcode model class names), the `datasets` library for
data, and the `Trainer` API for fine-tuning unless you need a custom loop.

## Essential Core

Load a pretrained model with its processor/tokenizer via `Auto*` classes, pinning `revision`:

```python
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", revision="main")
model = AutoModelForImageClassification.from_pretrained(
    "microsoft/resnet-50",
    revision="main",
    torch_dtype=torch.float32,
    cache_dir=None,          # set to share a cache across runs
    trust_remote_code=False, # only True for trusted sources
)
```

For quick inference, skip the manual wiring and use a task `pipeline`:

```python
from transformers import pipeline

classifier = pipeline("image-classification", model="microsoft/resnet-50", device=0)
results = classifier("path/to/image.jpg")
# -> [{"label": "cat", "score": 0.97}, ...]
```

Load data with `datasets` and map preprocessing in batches:

```python
from datasets import load_dataset

dataset = load_dataset("cifar10")
dataset = dataset.map(
    lambda batch: processor(images=batch["img"], return_tensors="pt"),
    batched=True,
    batch_size=32,
)
dataset.set_format("torch")
```

Fine-tune with `Trainer` rather than a hand-rolled loop unless you need custom behaviour:

```python
from transformers import Trainer, TrainingArguments

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="outputs/finetuned",
        num_train_epochs=10,
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["tensorboard"],
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("outputs/finetuned/best")
```

## Which Reference for Which Task

| Task | Entry point | Detail in |
|------|-------------|-----------|
| Load a model | `AutoModelFor*.from_pretrained` | `references/transformers-models.md` |
| Load/preprocess data | `load_dataset`, `.map` | `references/datasets.md` |
| Full fine-tuning run | `Trainer`, `TrainingArguments` | `references/fine-tuning-with-trainer.md` |
| Cheap fine-tuning | `LoraConfig`, `get_peft_model` | `references/peft-lora.md` |
| Share a model | `push_to_hub`, `ModelCard` | `references/hub-publishing.md` |
| Fast/small inference | `pipeline`, `BitsAndBytesConfig` | `references/inference-optimization.md` |

## Conventions

- Use `AutoModel*`, `AutoTokenizer`, `AutoImageProcessor` — never concrete class names.
- Pin `revision` on every `from_pretrained` call for reproducibility.
- Load models once at initialization; pass a `cache_dir` so repeated runs do not re-download.
- Preprocess with `dataset.map(..., batched=True)` and `set_format("torch")`.
- Drive `TrainingArguments` from a frozen Pydantic config so runs are typed and serializable.
- Prefer PEFT/LoRA over full fine-tuning when data is limited.
- Publish with a model card that carries license, dataset, and metrics.
- Log with Loguru; route Trainer telemetry with `report_to=["tensorboard"]` or `["wandb"]`.

## Anti-Patterns

- **Never hardcode model class names** — use `AutoModel`, `AutoTokenizer`, `AutoImageProcessor` for flexibility.
- **Never download models inside training loops** — load once at initialization, cache with `cache_dir`.
- **Never skip `revision` parameter** — pin model versions for reproducibility.
- **Never tokenize the full dataset eagerly** — use `dataset.map(batched=True)` with lazy loading.
- **Never fine-tune all parameters when data is limited** — use PEFT/LoRA to reduce overfitting.
- **Never push models to public hub without a model card** — always include metrics, dataset, and license info.
- **Never ignore `trust_remote_code` warnings** — only set to `True` for trusted model sources.

## Integration with Other Skills

Wrap HF models in a Lightning Trainer for custom loops; set `report_to=["wandb"]` for tracking;
export via `optimum`/ONNX for inference; deploy on SageMaker (HF DLC) or behind FastAPI.

## Deep dives

- `references/transformers-models.md` — read when loading vision or text models with a typed config, or wiring `device_map`/`torch_dtype`.
- `references/datasets.md` — read when loading, preprocessing, or tokenizing datasets, or building one from a local image folder.
- `references/fine-tuning-with-trainer.md` — read when setting up `TrainingArguments`, `compute_metrics`, or a full `Trainer` run with early stopping.
- `references/peft-lora.md` — read when applying LoRA adapters or reloading a saved adapter onto a base model.
- `references/hub-publishing.md` — read when pushing a model to the Hub or generating a model card.
- `references/inference-optimization.md` — read when building task pipelines or loading 4-bit/8-bit quantized models.

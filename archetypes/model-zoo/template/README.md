# ${project_name}

${description}

A curated collection of pretrained models. Every model is described by a
version-controlled YAML **model card**, its weights are fetched only after their
**SHA-256 digest is verified**, and any callable can be measured with the
built-in latency/throughput **benchmark harness**.

The core — cards, registry, download verification, benchmarking — is pure
Python. PyTorch is an *optional* extra, so the project is runnable and testable
on any machine, with or without a GPU.

## Layout

```
${project_slug}/
├── registry/                  # Model cards (version controlled, one YAML per model)
│   ├── resnet50.yaml
│   └── yolov8n.yaml
├── src/${package_name}/
│   ├── __init__.py            # Public API
│   ├── __main__.py            # CLI: list / show / download / verify
│   ├── model_card.py          # Pydantic V2 model-card schema
│   ├── registry.py            # Load, validate, and query the registry
│   ├── download.py            # SHA-256-verified download + local cache
│   ├── benchmark.py           # Latency / throughput harness
│   └── py.typed
├── tests/
├── pixi.toml
└── pyproject.toml
```

## Setup

```bash
pixi install
```

Add a dependency with `pixi add <package>` (or `pixi add --pypi <package>` for
PyPI-only packages).

## Usage

### From the command line

```bash
# List everything in the registry
python -m ${package_name} list

# Filter by task, tag, or status
python -m ${package_name} list --task detection
python -m ${package_name} list --tag imagenet

# Print a full model card
python -m ${package_name} show resnet50

# Download weights into the cache — refuses to keep the file on SHA-256 mismatch
python -m ${package_name} download resnet50

# Check a file you already have against the card
python -m ${package_name} verify resnet50 ./resnet50-v1.pth
```

### From Python

```python
from loguru import logger

from ${package_name} import download_weights, load_registry, run_benchmark

registry = load_registry("registry")

for card in registry.select(task="classification", metric="top1_accuracy", min_value=0.75):
    logger.info("{} top1={}", card.key, card.metric("top1_accuracy"))

card = registry.get("resnet50")  # highest version
card = registry.get("yolov8n", "1.2.0")  # pinned version

weights_path = download_weights(card)  # verified, cached, returns a Path

result = run_benchmark(lambda: my_session.run(None, feeds), name=card.key)
logger.info(result.summary())
```

## Model card schema

One YAML file per model under `registry/`. Unknown keys are rejected, so a typo
fails at load time rather than at inference time.

```yaml
name: resnet50                  # lowercase registry key
version: "1.0.0"                # dotted numeric version
task: classification            # classification | detection | segmentation | keypoint | embedding
architecture: resnet50
description: >-
  ResNet-50 image classifier trained on ImageNet-1K.
license: Apache-2.0             # SPDX identifier
training_dataset: ImageNet-1K
num_parameters: 25557032

inputs:
  channels: 3
  height: 224
  width: 224
  dtype: float32                # float32 | uint8
  mean: [0.485, 0.456, 0.406]   # length must equal `channels`
  std: [0.229, 0.224, 0.225]

weights:
  url: https://models.example.com/resnet50/v1.0.0/resnet50-v1.pth   # http(s):// or file://
  sha256: 3d05ac90a44862815a5c0fae3a4e1ee3b8e8e29bcb5c8c06459c6e865b342235
  size_bytes: 102530333
  weight_format: pytorch_state_dict   # pytorch_state_dict | torchscript | onnx | safetensors

evaluations:                    # optional, repeatable
  - dataset: ImageNet-1K
    split: val
    metrics:
      top1_accuracy: 0.7613
      top5_accuracy: 0.9290
    hardware: NVIDIA A100 40GB

tags: [imagenet, classification, baseline]
status: active                  # active | experimental | deprecated
```

> The URLs and digests in the shipped cards are placeholders. Replace them with
> your own artefacts and their real digests: `shasum -a 256 <file>`.

## Adding a model

1. Upload the weight file somewhere reachable over HTTPS.
2. Compute its digest: `shasum -a 256 my-model.pth`.
3. Copy `registry/resnet50.yaml` to `registry/<name>.yaml` and fill it in.
4. Run `pytest` — the card is validated by the schema tests automatically.
5. Confirm the fetch path: `python -m ${package_name} download <name>`.

## Weight cache

Downloads land in `.cache/weights/<name>/<version>/<filename>`. Override the
root with the `MODEL_WEIGHTS_CACHE` environment variable (`pixi.toml` sets it
for the project environment). The registry directory is likewise configurable
via `MODEL_REGISTRY_DIR`.

A cached file is re-hashed on every access. If it does not match the card, it is
re-downloaded; if the fresh download does not match either, `download_weights`
raises `ChecksumMismatchError` and nothing is written to the cache.

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Or via pixi tasks: `pixi run test`, `pixi run lint`, `pixi run typecheck`,
`pixi run quality`.

## Conventions

- **Loguru** for all logging — no `print`, no `logging.getLogger`.
- **Pydantic V2** for the model-card schema and every config object.
- **src-layout** with `py.typed`; `mypy --strict` must pass.
- Weight binaries are never committed — only their URL and SHA-256.

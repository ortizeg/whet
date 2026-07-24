# Model Zoo Archetype

A project template for curating a collection of pretrained computer vision models. Every model is described by a version-controlled YAML **model card**, its weights are fetched only after their **SHA-256 digest is verified**, and any callable can be measured with a small **latency/throughput benchmark harness**.

## Purpose

As ML teams mature, they accumulate a growing collection of pretrained models: baseline classifiers, fine-tuned detectors, experimental architectures, and production-deployed models. Without a structured system, these models exist as scattered checkpoint files on shared drives, unnamed weights in experiment tracking platforms, and undocumented artifacts on team members' local machines. Critical information such as training data provenance, input specifications, and benchmark results is lost or scattered across chat messages and wiki pages.

The Model Zoo archetype solves this by providing a centralized, version-controlled registry where every model is documented with a standardized model card and its weights are only ever loaded after a checksum match. The registry lives in git, so model metadata reviews go through the same pull-request process as code.

This archetype is not a training framework. It consumes trained checkpoints produced by training projects and packages them into a curated collection that the rest of the organization can discover and deploy with confidence.

## What the template actually generates

The core is **pure Python** — Pydantic V2, PyYAML, Loguru, and httpx. PyTorch is an optional extra, so a freshly generated project's tests pass on any machine, with or without a GPU.

```
${project_slug}/
├── .gitignore
├── README.md
├── pixi.toml                       # Canonical environment + tasks
├── pyproject.toml                  # Metadata, deps, ruff / mypy / pytest config
├── registry/                       # Model cards (version controlled)
│   ├── resnet50.yaml               # Example: classification
│   └── yolov8n.yaml                # Example: detection
├── src/${package_name}/
│   ├── __init__.py                 # Public API re-exports
│   ├── __main__.py                 # CLI: list / show / download / verify
│   ├── model_card.py               # Pydantic V2 model-card schema
│   ├── registry.py                 # Load, validate, and query model cards
│   ├── download.py                 # SHA-256-verified download + local cache
│   ├── benchmark.py                # Latency / throughput harness
│   └── py.typed
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Offline fixtures (file:// weights)
    ├── test_model_card.py          # Schema accepts good cards, rejects bad ones
    ├── test_registry.py            # Loading, lookup, versioning, filtering
    ├── test_download.py            # Checksum verification, including rejection
    ├── test_benchmark.py           # Harness works on any callable
    └── test_cli.py                 # Exit codes for each subcommand
```

## Key Features

- **Strict model-card schema** — Pydantic V2 with `extra="forbid"` and `frozen=True`, so a typo in a YAML card fails at load time with the offending path, not at inference time.
- **SHA-256-verified downloads** — weights are streamed to a `.part` file, hashed as they arrive, and only promoted into the cache on a digest match. A mismatch raises `ChecksumMismatchError` and leaves nothing behind.
- **Injectable fetchers** — the fetcher is a `Protocol`, and `file://` URLs are supported alongside `http(s)://`, so the whole download path is testable with no network access.
- **Self-healing cache** — a cached file is re-hashed on every access; a corrupt entry is transparently re-downloaded.
- **Version-aware lookup** — `registry.get("resnet50")` returns the highest version; `registry.get("resnet50", "1.0.0")` pins one.
- **Registry queries** — filter by task, tag, status, or a metric threshold.
- **Framework-agnostic benchmarking** — `run_benchmark` times any zero-argument callable, so the same harness measures a torch module, an ONNX Runtime session, or a stub.
- **CLI** — `python -m ${package_name} list | show | download | verify`, all output through Loguru.

## Model Card Schema

One YAML file per model under `registry/`, validated by `${package_name}.model_card.ModelCard`.

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
  std: [0.229, 0.224, 0.225]    # non-zero

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

The URLs and digests in the generated cards are placeholders. Replace them with your own artifacts and their real digests (`shasum -a 256 <file>`).

## Template Variables

`whet init` substitutes these into both file contents and file/directory names.

| Variable | Description | Default |
|---|---|---|
| `${project_name}` | Human-readable zoo name | Required |
| `${project_slug}` | Directory and distribution name | Derived from `project_name` |
| `${package_name}` | Python import name | Derived from `project_slug` |
| `${description}` | One-line project description | Empty |
| `${author}` | Maintainer name | Empty |
| `${python_version}` | Minimum Python version | 3.11 |

Runtime paths are configured with environment variables rather than template variables, so they can change without regenerating the project:

| Variable | Purpose | Default |
|---|---|---|
| `MODEL_REGISTRY_DIR` | Where model cards are discovered | `registry` |
| `MODEL_WEIGHTS_CACHE` | Where verified weights are cached | `.cache/weights` |

## Dependencies

Runtime (`[project].dependencies`):

```toml
pydantic = ">=2.6"
pyyaml = ">=6.0"
loguru = ">=0.7"
httpx = ">=0.27"
```

Dev extra: `pytest`, `pytest-cov`, `ruff`, `mypy`, `types-PyYAML`.
Optional `torch` extra: `torch`, `torchvision` — only needed to benchmark real PyTorch models.

## Usage

### Setup

```bash
pixi install
pixi add <package>          # conda-forge
pixi add --pypi <package>   # PyPI-only
```

### Command line

```bash
python -m ${package_name} list
python -m ${package_name} list --task detection
python -m ${package_name} list --tag imagenet
python -m ${package_name} show resnet50
python -m ${package_name} download resnet50
python -m ${package_name} verify resnet50 ./resnet50-v1.pth
```

`download` exits `2` and writes nothing to the cache when the digest does not match.

### Python API

```python
from loguru import logger

from ${package_name} import download_weights, load_registry, run_benchmark

registry = load_registry("registry")

for card in registry.select(task="classification", metric="top1_accuracy", min_value=0.75):
    logger.info("{} top1={}", card.key, card.metric("top1_accuracy"))

card = registry.get("resnet50")          # highest version
weights_path = download_weights(card)    # verified, cached, returns a Path

result = run_benchmark(lambda: session.run(None, feeds), name=card.key)
logger.info(result.summary())
```

### Quality gates

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Or `pixi run quality`, which chains all four.

## Adding a Model

1. Upload the weight file somewhere reachable over HTTPS.
2. Compute its digest: `shasum -a 256 my-model.pth`.
3. Copy `registry/resnet50.yaml` to `registry/<name>.yaml` and fill it in.
4. Run `pytest` — the schema test validates every card in `registry/` automatically.
5. Confirm the fetch path: `python -m ${package_name} download <name>`.

## Extension Points

The generated project is a working core, deliberately small. Common next steps, none of which are scaffolded:

- **Model loading** — add a `loader.py` that maps `card.architecture` to a constructor and calls `download_weights` before `load_state_dict`. Keep it behind the optional `torch` extra so the core stays importable without a deep-learning stack.
- **Export formats** — add ONNX / TorchScript / TensorRT export and record the resulting URL plus digest as a second `WeightsSpec` on the card.
- **More storage backends** — `download.py` dispatches on URL scheme in `default_fetcher`; add `s3://` or `gs://` by extending it and adding the scheme to `SUPPORTED_SCHEMES` in `model_card.py`.
- **Accuracy benchmarks** — `benchmark.py` measures latency only. Dataset-level metrics belong in a separate module; see the `model-evaluation` skill.
- **CI** — a workflow that runs `pytest` on every PR keeps malformed model cards out of the registry. See the `github-actions` skill.

## Related Skills

Required: `pydantic`, `loguru`, `testing`, `onnx`, `pytorch-lightning`.
Recommended: `model-evaluation`, `wandb`, `docker-cv`.

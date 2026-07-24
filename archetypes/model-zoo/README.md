# Model Zoo Archetype

A structured project template for managing, benchmarking, and serving collections of pretrained computer vision models. This archetype provides a unified registry for organizing multiple model architectures, a benchmark suite for systematic performance comparison, download and caching utilities for weight management, and export tools for deploying models across inference frameworks.

## Purpose

As ML teams mature, they accumulate a growing collection of pretrained models: baseline classifiers, fine-tuned detectors, experimental architectures, and production-deployed models. Without a structured system, these models exist as scattered checkpoint files on shared drives, unnamed weights in experiment tracking platforms, and undocumented artifacts on team members' local machines. Critical information such as training data provenance, hyperparameters, input specifications, and benchmark results is lost or scattered across Slack messages and wiki pages.

The Model Zoo archetype solves this by providing a centralized, version-controlled registry where every model is documented with a standardized model card, associated with reproducible benchmark results, and accessible through a simple download-and-load API. The registry tracks model metadata, performance metrics, export formats, and lineage information in a structured format that is both machine-readable and human-browsable.

This archetype is not a training framework. It consumes trained checkpoints produced by training projects and packages them into a curated collection that the rest of the organization can discover, evaluate, and deploy with confidence.

## Use Cases

- **Model benchmarking** -- Run systematic comparisons of multiple architectures on standardized datasets with consistent evaluation protocols and hardware conditions.
- **Ensemble inference** -- Load and combine predictions from multiple models with configurable weighting and fusion strategies.
- **Model selection** -- Query the registry by task, performance threshold, latency budget, or model size to find the best model for a given deployment constraint.
- **Architecture exploration** -- Evaluate a new architecture against established baselines with a single command.
- **Model versioning** -- Track model iterations over time with clear version semantics, deprecation notices, and migration guides.
- **Export and deployment** -- Convert models from PyTorch to ONNX, TorchScript, or TensorRT for deployment across different inference platforms.
- **Model auditing** -- Maintain a complete record of each model's training provenance, data dependencies, and evaluation history for compliance and reproducibility.

## Directory Structure

```
{{project_slug}}/
├── .github/
│   └── workflows/
│       ├── benchmark.yml              # Scheduled benchmark runs
│       ├── test.yml                   # Registry and loader tests
│       └── ci.yml                    # Lint, type check, tests
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml
├── README.md
├── registry/                          # Model registry (version controlled)
│   ├── index.yaml                    # Master index of all models
│   ├── classification/
│   │   ├── resnet50/
│   │   │   ├── model_card.yaml       # Model metadata and documentation
│   │   │   ├── config.yaml           # Architecture config
│   │   │   └── benchmarks.yaml       # Benchmark results
│   │   └── efficientnet_b0/
│   │       ├── model_card.yaml
│   │       ├── config.yaml
│   │       └── benchmarks.yaml
│   ├── detection/
│   │   └── yolov8/
│   │       ├── model_card.yaml
│   │       ├── config.yaml
│   │       └── benchmarks.yaml
│   └── segmentation/
│       └── unet/
│           ├── model_card.yaml
│           ├── config.yaml
│           └── benchmarks.yaml
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── py.typed
│   ├── registry.py                    # Model registry and discovery
│   ├── loader.py                      # Model loading and instantiation
│   ├── downloader.py                  # Weight download and caching
│   ├── config.py                      # Pydantic config models
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract model interface
│   │   ├── classification/
│   │   │   ├── __init__.py
│   │   │   ├── resnet.py
│   │   │   └── efficientnet.py
│   │   ├── detection/
│   │   │   ├── __init__.py
│   │   │   └── yolo.py
│   │   └── segmentation/
│   │       ├── __init__.py
│   │       └── unet.py
│   ├── benchmark/
│   │   ├── __init__.py
│   │   ├── runner.py                 # Benchmark orchestrator
│   │   ├── metrics.py                # Standardized evaluation metrics
│   │   ├── datasets.py               # Benchmark dataset loaders
│   │   └── report.py                 # Benchmark report generation
│   ├── export/
│   │   ├── __init__.py
│   │   ├── onnx.py                   # ONNX export
│   │   ├── torchscript.py           # TorchScript export
│   │   └── tensorrt.py              # TensorRT export (optional)
│   └── utils/
│       ├── __init__.py
│       ├── cache.py                  # Local cache management
│       ├── hashing.py               # Weight file verification
│       └── io.py                     # File I/O helpers
├── cache/                             # Local weight cache (gitignored)
│   └── .gitkeep
├── scripts/
│   ├── benchmark_all.py              # Run all benchmarks
│   ├── benchmark_model.py            # Benchmark a single model
│   ├── add_model.py                  # Interactive model registration
│   ├── export_model.py               # Export model to target format
│   └── download_weights.py           # Download model weights
├── benchmarks/
│   ├── results/                      # Benchmark result storage
│   │   └── .gitkeep
│   └── configs/
│       ├── imagenet_val.yaml         # ImageNet validation benchmark
│       ├── coco_val.yaml             # COCO validation benchmark
│       └── latency.yaml             # Latency benchmark config
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_registry.py
│   ├── test_loader.py
│   ├── test_downloader.py
│   ├── test_export.py
│   └── test_benchmark.py
└── docs/
    ├── model_card_template.md        # Template for new model cards
    └── contributing.md               # Guide for adding new models
```

## Key Features

- **Centralized model registry** with YAML-based model cards containing architecture, training provenance, input specifications, and performance metrics.
- **Version-controlled metadata** where all model documentation and benchmark results live in git, providing a complete audit trail.
- **Download and caching** utilities that fetch weights from remote storage (S3, GCS, HuggingFace Hub, HTTP URLs) with SHA256 verification and local caching.
- **Benchmark suite** with standardized evaluation protocols, dataset loaders, and automated report generation for fair model comparisons.
- **Multi-format export** to ONNX, TorchScript, and TensorRT with input/output shape validation and accuracy verification post-export.
- **Model discovery API** for querying the registry by task, performance, size, or custom metadata fields.
- **Weight integrity verification** through SHA256 checksums recorded in model cards and verified at download time.

## Model Card Schema

Every model in the registry has a `model_card.yaml` that follows a standardized schema.

```yaml
name: "ResNet-50"
version: "1.0.0"
task: "classification"
framework: "pytorch"
description: "ResNet-50 trained on ImageNet-1K with standard augmentation."

architecture:
  backbone: "resnet50"
  num_parameters: 25557032
  input_size: [3, 224, 224]
  input_dtype: "float32"
  input_normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

training:
  dataset: "ImageNet-1K"
  dataset_size: 1281167
  epochs: 90
  optimizer: "SGD"
  learning_rate: 0.1
  batch_size: 256
  hardware: "8x NVIDIA A100"
  training_time_hours: 24

weights:
  url: "https://storage.example.com/models/resnet50_v1.0.0.pth"
  sha256: "a1b2c3d4e5f6..."
  size_mb: 97.8
  format: "pytorch_state_dict"

performance:
  imagenet_val:
    top1_accuracy: 0.7613
    top5_accuracy: 0.9290
    inference_latency_ms: 4.2
    throughput_fps: 238.0
    hardware: "NVIDIA A100"
    batch_size: 1

exports:
  onnx:
    url: "https://storage.example.com/models/resnet50_v1.0.0.onnx"
    sha256: "f6e5d4c3b2a1..."
    opset_version: 17

tags: ["imagenet", "classification", "baseline"]
status: "active"  # active, deprecated, experimental
deprecated_by: null
```

## Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `{{project_name}}` | Human-readable zoo name | Required |
| `{{project_slug}}` | Directory name | Auto-generated |
| `{{package_name}}` | Python import name | Auto-generated |
| `{{author_name}}` | Maintainer name | Required |
| `{{email}}` | Maintainer email | Required |
| `{{description}}` | Zoo description | Required |
| `{{python_version}}` | Python version | 3.11 |
| `{{weight_storage}}` | Remote storage backend (s3, gcs, http) | s3 |
| `{{cache_dir}}` | Local weight cache directory | `~/.cache/{{package_name}}` |

## Dependencies

```toml
[dependencies]
python = ">=3.11"
torch = ">=2.0"
torchvision = ">=0.16"
onnx = ">=1.14"
onnxruntime = ">=1.17"
pydantic = ">=2.0"
pyyaml = ">=6.0"
rich = ">=13.0"
requests = ">=2.31"
tqdm = ">=4.66"
pandas = ">=2.1"
```

## Usage

### Listing Available Models

```bash
# List all models in the registry
pixi run python -m {{package_name}} list

# Filter by task
pixi run python -m {{package_name}} list --task detection

# Filter by performance threshold
pixi run python -m {{package_name}} list --task classification --min-accuracy 0.80
```

### Loading a Model

```python
from {{package_name}} import load_model, list_models

# Discover available models
models = list_models(task="classification")
for m in models:
    print(f"{m.name} v{m.version}: top1={m.performance.top1_accuracy:.3f}")

# Load a model with pretrained weights (downloads and caches automatically)
model = load_model("resnet50", pretrained=True)
model.eval()

# Load a specific version
model = load_model("resnet50", version="1.0.0", pretrained=True)

# Load without pretrained weights (architecture only)
model = load_model("resnet50", pretrained=False, num_classes=10)
```

### Running Benchmarks

```bash
# Benchmark all models on ImageNet validation
pixi run python scripts/benchmark_all.py --config benchmarks/configs/imagenet_val.yaml

# Benchmark a single model
pixi run python scripts/benchmark_model.py --model resnet50 --config benchmarks/configs/imagenet_val.yaml

# Run latency benchmarks
pixi run python scripts/benchmark_model.py --model resnet50 --config benchmarks/configs/latency.yaml

# Generate a comparison report
pixi run python -m {{package_name}}.benchmark.report --results benchmarks/results/ --output reports/comparison.html
```

### Exporting Models

```bash
# Export to ONNX
pixi run python scripts/export_model.py --model resnet50 --format onnx --output exports/

# Export to TorchScript
pixi run python scripts/export_model.py --model resnet50 --format torchscript --output exports/

# Export with accuracy verification
pixi run python scripts/export_model.py --model resnet50 --format onnx --verify --num-samples 100
```

### Adding a New Model

```bash
# Interactive model registration
pixi run python scripts/add_model.py

# This will:
# 1. Prompt for model metadata (name, task, architecture)
# 2. Create the registry directory structure
# 3. Generate a model_card.yaml template
# 4. Run initial benchmarks if weights are available
```

## Customization Guide

### Adding a New Model Architecture

1. Implement the model class in `src/{{package_name}}/models/<task>/` inheriting from the base model interface.
2. Create a registry directory under `registry/<task>/<model_name>/` with `model_card.yaml`, `config.yaml`, and `benchmarks.yaml`.
3. Register the model in `registry/index.yaml`.
4. Upload weights to the configured remote storage backend.
5. Record the weight URL and SHA256 checksum in the model card.
6. Run benchmarks and update `benchmarks.yaml` with results.

### Adding a New Benchmark Dataset

1. Create a dataset loader in `src/{{package_name}}/benchmark/datasets.py` that returns a standard `torch.utils.data.Dataset`.
2. Create a benchmark configuration YAML in `benchmarks/configs/` specifying the dataset, metrics, and evaluation protocol.
3. Add task-specific metrics to `src/{{package_name}}/benchmark/metrics.py` if needed.

### Adding a New Export Format

1. Create an export module in `src/{{package_name}}/export/` implementing the export interface.
2. The export function should accept a loaded PyTorch model, example inputs, and output path.
3. Include post-export verification that runs inference on sample inputs and compares outputs against the PyTorch reference.
4. Add the export URL and checksum fields to the model card schema.

### Custom Weight Storage Backends

The `downloader.py` module supports pluggable storage backends. To add a new backend (e.g., Azure Blob Storage), implement the `StorageBackend` interface with `download(url, local_path)` and `exists(url)` methods, then register it in the backend factory. The default backends support S3 (via boto3), GCS (via google-cloud-storage), HuggingFace Hub, and plain HTTP/HTTPS URLs.

### Cache Management

The local weight cache stores downloaded model files in `{{cache_dir}}` organized by model name and version. Use the cache management utilities to inspect cache contents, calculate total size, and evict old versions. The cache respects SHA256 checksums: if a cached file does not match the expected checksum from the model card, it is re-downloaded automatically.

```bash
# Show cache contents and size
pixi run python -m {{package_name}}.utils.cache info

# Clear cache for a specific model
pixi run python -m {{package_name}}.utils.cache clear --model resnet50

# Clear entire cache
pixi run python -m {{package_name}}.utils.cache clear --all
```

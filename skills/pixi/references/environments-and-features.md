# Pixi Features and Environments

Scope: organizing optional dependencies into named features (dev, docs, cuda, onnx,
wandb) and composing those features into installable environments.

## Feature Groups

Features let you organize optional dependencies into named groups. This keeps the default
environment lean while allowing developers to opt into additional capabilities.

### Defining Features

```toml
# Core dependencies (always installed)
[dependencies]
python = ">=3.11"
numpy = ">=1.26"

# Development tools (installed in dev environments)
[feature.dev.dependencies]
pytest = ">=7.4"
ruff = ">=0.8"
mypy = ">=1.11"

# Documentation tools
[feature.docs.dependencies]
mkdocs = ">=1.5"
mkdocstrings = ">=0.24"

# CUDA-specific dependencies
[feature.cuda.dependencies]
cuda-toolkit = ">=12.1"

[feature.cuda.pypi-dependencies]
torch = { version = ">=2.2", extras = ["cuda"] }

# ONNX export dependencies
[feature.onnx.pypi-dependencies]
onnx = ">=1.15"
onnxruntime = ">=1.17"

# Experiment tracking
[feature.wandb.pypi-dependencies]
wandb = ">=0.16"
```

## Composing Environments from Features

```toml
[environments]
# Default: core + dev tools
default = { features = ["dev"], solve-group = "default" }

# Training: core + dev + cuda + wandb
train = { features = ["dev", "cuda", "wandb"], solve-group = "default" }

# CI: core + dev (no cuda, no wandb)
ci = { features = ["dev"], solve-group = "default" }

# Docs: core + dev + docs
docs = { features = ["dev", "docs"], solve-group = "default" }

# Export: core + dev + onnx
export = { features = ["dev", "onnx"], solve-group = "default" }
```

## Using Environments

```bash
# Install default environment
pixi install

# Install specific environment
pixi install -e train

# Run task in specific environment
pixi run -e train train
pixi run -e docs docs-serve
pixi run -e export export-onnx
```

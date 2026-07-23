# Project-Type Specific pixi.toml Templates

Scope: complete, known-good `pixi.toml` manifests for the three common project shapes —
training project, inference service, and library package.

## Contents

- [Training Project pixi.toml](#training-project-pixitoml)
- [Inference Service pixi.toml](#inference-service-pixitoml)
- [Library Package pixi.toml](#library-package-pixitoml)

## Training Project pixi.toml

```toml
[project]
name = "training-project"
channels = ["conda-forge", "pytorch"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"
pillow = ">=10.0"

[pypi-dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
lightning = ">=2.2"
albumentations = ">=1.3"
pydantic = ">=2.6"
wandb = ">=0.16"
timm = ">=0.9"
rich = ">=13.0"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[tasks]
train = "python -m training_project.train"
eval = "python -m training_project.evaluate"
test = "pytest tests/ -v --cov=src --cov-fail-under=80"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
```

## Inference Service pixi.toml

```toml
[project]
name = "inference-service"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"

[pypi-dependencies]
torch = ">=2.2"
torchvision = ">=0.17"
fastapi = ">=0.109"
uvicorn = { version = ">=0.27", extras = ["standard"] }
pydantic = ">=2.6"
onnxruntime = ">=1.17"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
httpx = ">=0.27"

[tasks]
serve = "uvicorn inference_service.api:app --host 0.0.0.0 --port 8000"
serve-dev = "uvicorn inference_service.api:app --host 0.0.0.0 --port 8000 --reload"
test = "pytest tests/ -v"
lint = "ruff check ."
typecheck = "mypy src/ --strict"
```

## Library Package pixi.toml

```toml
[project]
name = "cv-library"
channels = ["conda-forge"]
platforms = ["linux-64", "osx-arm64", "osx-64"]

[dependencies]
python = ">=3.11"
numpy = ">=1.26"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
pre-commit = ">=3.5"

[feature.docs.dependencies]
mkdocs = ">=1.5"
mkdocstrings = ">=0.24"
mkdocs-material = ">=9.5"

[tasks]
test = "pytest tests/ -v --cov=src --cov-report=term --cov-fail-under=90"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
docs = "mkdocs serve"
docs-build = "mkdocs build"
```

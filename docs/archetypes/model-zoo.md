# Model Zoo

Collection of pretrained models with standardized interfaces, model cards, download management, and benchmarking.

## Purpose

This archetype manages a collection of pretrained models with a unified API. Each model has a model card documenting its architecture, training data, performance metrics, and limitations. The zoo provides a consistent interface for loading, running inference, and benchmarking models, making it easy to compare alternatives and swap implementations.

## Directory Structure

```
{{project_slug}}/
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── registry.py            # Model registry
│   ├── base.py                # Abstract model interface
│   ├── download.py            # Weight download manager
│   ├── benchmark.py           # Benchmarking utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet/
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   └── MODEL_CARD.md
│   │   └── efficientnet/
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── MODEL_CARD.md
│   └── configs/
│       ├── __init__.py
│       └── model_configs.py   # Pydantic model configs
├── benchmarks/
│   ├── results/
│   └── run_benchmarks.py
├── tests/
│   ├── test_registry.py
│   ├── test_download.py
│   └── test_models.py
└── ...
```

## Registry API

```python
from my_project import load_model

# Load a model by name
model = load_model("resnet50", pretrained=True)

# List available models
from my_project import list_models
print(list_models())  # ["resnet50", "efficientnet_b0", ...]

# Get model card
from my_project import get_model_card
card = get_model_card("resnet50")
```

## Model Card Contents

Each model includes a `MODEL_CARD.md` documenting:

- Architecture description and diagram
- Training data and preprocessing
- Performance metrics (accuracy, FPS, model size)
- Known limitations and biases
- Citation and license

## Usage

```bash
# Run benchmarks
python benchmarks/run_benchmarks.py

# Download all model weights
python -m my_project.download --all

# Compare models
python -m my_project.benchmark --models resnet50,efficientnet_b0
```

## Customization

- Add new models in `src/{{package_name}}/models/`
- Register models using the `@MODELS.register("name")` decorator
- Create model cards following the template
- Add benchmarking datasets to `benchmarks/`

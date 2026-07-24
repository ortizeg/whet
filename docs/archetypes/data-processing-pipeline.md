# Data Processing Pipeline

ETL workflows for dataset preparation, transformation, and validation with parallel processing and dataset versioning.

## Purpose

This archetype structures dataset processing as a series of well-defined stages: download, preprocess, validate, split, and package. Each stage is a self-contained module with Pydantic-validated configuration, making pipelines reproducible and composable. Large data assets live in object storage and are tracked by a content-hashed manifest, and parallel processing handles large-scale datasets efficiently.

## Directory Structure

```
${project_slug}/
├── conf/
│   └── pipeline.toml
├── src/
│   └── ${package_name}/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── manifest.py
│       ├── pipeline.py
│       ├── py.typed
│       ├── quality.py
│       ├── sample_data.py
│       ├── splitting.py
│       └── stages.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_manifest.py
│   ├── test_pipeline.py
│   ├── test_quality.py
│   └── test_splitting.py
├── .gitignore
├── README.md
├── pixi.toml
└── pyproject.toml
```

## Pipeline Stage Interface

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class StageConfig(BaseModel):
    """Base configuration for pipeline stages."""
    input_dir: str
    output_dir: str

class Stage(ABC):
    @abstractmethod
    def run(self, config: StageConfig) -> None: ...
    @abstractmethod
    def validate(self) -> bool: ...
```

## Usage

```bash
# Run full pipeline
python -m my_project.pipeline

# Run single stage
python -m my_project.pipeline stage=preprocess
```

## Customization

- Add new stages in `src/{{package_name}}/stages/`
- Define stage configs in `configs/stages/`
- Add image transforms in `transforms/`
- Configure the object storage bucket and prefix used for dataset artifacts

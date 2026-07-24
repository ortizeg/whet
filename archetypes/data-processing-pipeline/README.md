# Data Processing Pipeline Archetype

A structured project template for building robust ETL (Extract, Transform, Load) workflows for machine learning datasets. This archetype provides a pipeline framework for cleaning, transforming, validating, augmenting, and splitting datasets with full traceability, parallel processing support, and integration with data versioning tools.

## Purpose

Data preparation is the most time-consuming and error-prone phase of any machine learning project. Raw datasets arrive in inconsistent formats, contain corrupt files, have labeling errors, and require extensive transformation before they are suitable for training. Despite this, data processing code is often the least structured part of an ML codebase -- scattered across ad-hoc scripts, undocumented Jupyter cells, and bash one-liners that are impossible to reproduce.

The Data Processing Pipeline archetype solves this by providing a modular, testable, and reproducible framework for dataset preparation. Each processing step is implemented as an isolated stage with defined inputs, outputs, and validation criteria. Stages are composed into pipelines that can be executed end-to-end or incrementally, with each intermediate result cached and validated. The framework supports parallel processing for throughput-intensive operations and versions datasets by keeping the bulk data in object storage alongside a content-hashed manifest committed to the repository.

The core design principle is that every transformation applied to data must be explicit, tested, logged, and reversible. This ensures that when model performance changes, the data lineage can be traced back to identify whether the change originated in the data pipeline or the model.

## Use Cases

- **Dataset cleaning** -- Remove corrupt images, fix encoding issues, standardize file formats, and handle missing annotations.
- **Annotation format conversion** -- Convert between COCO, VOC, YOLO, and custom annotation formats with validation.
- **Image preprocessing** -- Resize, crop, normalize, and color-correct images to consistent specifications.
- **Data augmentation** -- Generate augmented training samples with controlled augmentation policies and deduplication.
- **Train/val/test splitting** -- Create reproducible dataset splits with stratification, group awareness, and cross-validation fold generation.
- **Dataset merging** -- Combine multiple source datasets with label harmonization and conflict resolution.
- **Quality assurance** -- Run automated checks for class balance, image quality, annotation consistency, and data leakage between splits.
- **Feature extraction** -- Pre-compute embeddings, feature maps, or derived features and store them for efficient training.

## Directory Structure

```
{{project_slug}}/
├── .github/
│   └── workflows/
│       ├── test.yml                    # Pipeline test suite
│       └── ci.yml                     # Lint, type check, tests
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml
├── README.md
├── params.yaml                         # Pipeline parameters
├── conf/
│   ├── pipeline.yaml                  # Pipeline stage configuration
│   ├── cleaning/
│   │   └── default.yaml
│   ├── splitting/
│   │   └── default.yaml
│   └── augmentation/
│       └── default.yaml
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── py.typed
│   ├── config.py                       # Pydantic pipeline config
│   ├── pipeline.py                     # Pipeline orchestrator
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── base.py                    # Abstract stage interface
│   │   ├── ingest.py                  # Data ingestion stage
│   │   ├── validate.py               # Data validation stage
│   │   ├── clean.py                   # Data cleaning stage
│   │   ├── transform.py              # Transformation stage
│   │   ├── augment.py                # Augmentation stage
│   │   ├── split.py                  # Train/val/test splitting
│   │   └── export.py                 # Output format export
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── image.py                   # Image integrity checks
│   │   ├── annotation.py             # Annotation consistency checks
│   │   └── schema.py                 # Data schema validation
│   ├── io/
│   │   ├── __init__.py
│   │   ├── readers.py                # Format-specific readers
│   │   ├── writers.py                # Format-specific writers
│   │   └── formats.py                # Format definitions
│   └── utils/
│       ├── __init__.py
│       ├── parallel.py               # Multiprocessing utilities
│       ├── hashing.py                # Content hashing for dedup
│       └── progress.py               # Progress bar helpers
├── data/
│   ├── raw/                           # Original source data
│   │   └── .gitkeep
│   ├── interim/                       # Intermediate stage outputs
│   │   └── .gitkeep
│   ├── processed/                     # Final processed dataset
│   │   └── .gitkeep
│   └── external/                      # Third-party reference data
│       └── .gitkeep
├── scripts/
│   ├── run_pipeline.py               # Full pipeline execution
│   ├── run_stage.py                  # Single stage execution
│   └── validate_dataset.py           # Standalone validation
├── reports/
│   ├── .gitkeep
│   └── templates/
│       └── data_report.html          # Dataset report template
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_stages.py
│   ├── test_validators.py
│   ├── test_io.py
│   └── fixtures/
│       └── sample_data/
│           └── .gitkeep
└── notebooks/
    └── data_exploration.ipynb         # Interactive data inspection
```

## Key Features

- **Modular stage architecture** where each processing step is an isolated, testable unit with defined inputs, outputs, and validation criteria.
- **Pipeline orchestration** that composes stages into reproducible end-to-end workflows with dependency resolution and caching.
- **Pydantic validation** at every stage boundary to catch data quality issues early and provide clear error messages.
- **Parallel processing** with configurable worker pools for CPU-bound operations like image resizing and augmentation.
- **Progress tracking** with rich progress bars showing per-stage and overall pipeline completion.
- **Content hashing** for deduplication and change detection, enabling incremental processing of modified files only.
- **Dataset versioning** through content-hashed manifests kept in Git while the bulk data lives in object storage, so pipeline runs reproduce with exact data lineage.
- **HTML reports** generated after each pipeline run summarizing dataset statistics, quality metrics, and processing logs.

## Pipeline Stage Interface

Every stage implements a common interface defined in `stages/base.py`.

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class StageConfig(BaseModel):
    """Configuration for a pipeline stage."""
    enabled: bool = True
    num_workers: int = 4

class PipelineStage(ABC):
    """Abstract base class for all pipeline stages."""

    @abstractmethod
    def run(self, input_path: Path, output_path: Path, config: StageConfig) -> StageResult:
        """Execute the stage, reading from input_path and writing to output_path."""
        ...

    @abstractmethod
    def validate_input(self, input_path: Path) -> ValidationResult:
        """Validate that input data meets this stage's requirements."""
        ...

    @abstractmethod
    def validate_output(self, output_path: Path) -> ValidationResult:
        """Validate that output data meets expected quality criteria."""
        ...
```

## Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `{{project_name}}` | Human-readable project name | Required |
| `{{project_slug}}` | Directory name | Auto-generated |
| `{{package_name}}` | Python import name | Auto-generated |
| `{{author_name}}` | Author name | Required |
| `{{email}}` | Author email | Required |
| `{{description}}` | Pipeline purpose description | Required |
| `{{python_version}}` | Python version | 3.11 |
| `{{input_format}}` | Source data format (coco, voc, yolo, custom) | coco |
| `{{output_format}}` | Target data format | coco |

## Dependencies

```toml
[dependencies]
python = ">=3.11"
pydantic = ">=2.0"
pyyaml = ">=6.0"
pillow = ">=10.0"
numpy = ">=1.26"
pandas = ">=2.1"
albumentations = ">=1.3"
tqdm = ">=4.66"
rich = ">=13.0"
pyarrow = ">=14.0"
```

## Usage

### Running the Full Pipeline

```bash
# Install dependencies
pixi install

# Place raw data in data/raw/
cp -r /path/to/source/dataset/* data/raw/

# Run the full pipeline
pixi run python scripts/run_pipeline.py

# Run with custom configuration overrides
pixi run python scripts/run_pipeline.py --config conf/pipeline.yaml \
    --override splitting.test_ratio=0.15
```

### Running Individual Stages

```bash
# Run only the validation stage
pixi run python scripts/run_stage.py validate --input data/raw/ --output reports/

# Run only the cleaning stage
pixi run python scripts/run_stage.py clean --input data/raw/ --output data/interim/cleaned/

# Run only the splitting stage
pixi run python scripts/run_stage.py split --input data/interim/cleaned/ --output data/processed/
```

### Validation

```bash
# Validate a dataset independently
pixi run python scripts/validate_dataset.py data/processed/ --format coco --report reports/validation.html

# Check for data leakage between splits
pixi run python scripts/validate_dataset.py data/processed/ --check-leakage
```

## Customization Guide

### Adding a New Pipeline Stage

1. Create a new class inheriting from `PipelineStage` in `src/{{package_name}}/stages/`.
2. Implement `run()`, `validate_input()`, and `validate_output()` methods.
3. Define a `StageConfig` subclass with Pydantic-validated parameters.
4. Add the stage configuration to `conf/pipeline.yaml`.
5. Register the stage in the pipeline orchestrator's stage registry.
6. Write unit tests in `tests/test_stages.py` with fixture data.

### Adding a New Data Format

1. Implement a reader in `src/{{package_name}}/io/readers.py` that parses the format into the internal representation.
2. Implement a writer in `src/{{package_name}}/io/writers.py` that serializes the internal representation to the target format.
3. Register the format in `src/{{package_name}}/io/formats.py`.
4. Add validation rules in `src/{{package_name}}/validators/schema.py`.

### Parallel Processing Configuration

The `parallel.py` utility module provides a `parallel_map` function that distributes work across a configurable number of processes. Each stage can specify its own `num_workers` parameter. For I/O-bound stages (downloading, reading from network storage), use thread-based parallelism. For CPU-bound stages (image resizing, augmentation), use process-based parallelism. The default is process-based with `num_workers=4`.

### Custom Validation Rules

Add domain-specific validation rules in `src/{{package_name}}/validators/`. Common additions include minimum image resolution checks, aspect ratio constraints, bounding box sanity checks (non-zero area, within image bounds), and class label consistency verification. Each validator returns a structured `ValidationResult` with severity levels (error, warning, info) and per-sample details.

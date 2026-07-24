# Research Notebook Archetype

A structured Jupyter-based project template for rapid experimentation, prototyping, and research exploration in computer vision and machine learning. This archetype provides an organized framework for exploratory data analysis, model prototyping, paper reproduction, and experiment documentation while maintaining enough structure to graduate promising experiments into production code.

## Purpose

Research work in machine learning is inherently iterative and exploratory. Notebooks are the natural medium for this work because they combine code execution, visualization, and narrative documentation in a single artifact. However, unstructured notebook projects quickly devolve into a tangle of unnamed files, hidden state dependencies, and unreproducible results.

The Research Notebook archetype solves this by imposing lightweight conventions on notebook organization, naming, and data management without sacrificing the flexibility that makes notebooks valuable. It includes utility modules for common operations, integration points for experiment tracking, and a clear pathway for promoting successful experiments to standalone scripts or full training projects.

Every notebook in this archetype follows a numbered naming convention, includes a purpose statement in its first cell, and declares its dependencies explicitly. Shared utility code lives in importable Python modules rather than being copy-pasted between notebooks, and data paths are managed through a centralized configuration rather than hardcoded strings.

## Use Cases

- **Exploratory data analysis (EDA)** -- Investigate dataset statistics, class distributions, annotation quality, and outliers before committing to a training approach.
- **Model exploration and prototyping** -- Quickly test model architectures, loss functions, and training strategies with small data subsets before scaling up.
- **Paper reproduction** -- Reproduce published results in a documented, shareable format with clear methodology notes.
- **Visualization and reporting** -- Generate publication-quality figures, confusion matrices, and performance plots for stakeholder communication.
- **Ablation studies** -- Systematically vary components of a pipeline and document the impact of each change.
- **Dataset curation** -- Interactively inspect, filter, relabel, and augment training data with visual feedback.

## Directory Structure

```
{{project_slug}}/
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Dataset overview and statistics
│   ├── 02_preprocessing_test.ipynb     # Augmentation and transform tests
│   ├── 03_baseline_model.ipynb         # First model prototype
│   ├── 04_experiment_tracking.ipynb    # Tracked experiment runs
│   └── _template.ipynb                 # Blank template with standard cells
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── config.py                       # Centralized paths and settings
│   ├── plotting.py                     # Reusable plotting functions
│   ├── data_utils.py                   # Data loading helpers
│   ├── model_utils.py                  # Model construction helpers
│   └── eval_utils.py                   # Evaluation and metric helpers
├── data/
│   ├── raw/                            # Original unmodified data
│   │   └── .gitkeep
│   ├── processed/                      # Cleaned and transformed data
│   │   └── .gitkeep
│   └── external/                       # Third-party datasets
│       └── .gitkeep
├── outputs/
│   ├── figures/                        # Saved plots and visualizations
│   │   └── .gitkeep
│   ├── models/                         # Saved checkpoints
│   │   └── .gitkeep
│   └── reports/                        # Generated reports
│       └── .gitkeep
├── scripts/
│   ├── setup_data.py                   # Download and prepare data
│   └── export_notebook.py             # Convert notebook to script
├── references/
│   └── .gitkeep                        # Papers, links, notes
└── tests/
    ├── __init__.py
    └── test_utils.py                   # Tests for utility modules
```

## Key Features

- **Numbered notebook convention** for clear execution order and logical progression through an analysis.
- **Notebook template** with standardized header cells for purpose, author, date, and dependency declarations.
- **Importable utility modules** that keep notebooks focused on analysis rather than boilerplate code.
- **Centralized configuration** for data paths, output directories, and experiment parameters.
- **Data directory structure** that separates raw, processed, and external data with gitkeep placeholders.
- **Output management** with dedicated directories for figures, model checkpoints, and generated reports.
- **Export pipeline** for converting validated notebooks into standalone Python scripts.
- **Pre-commit hooks** with notebook-specific linting and output stripping.

## Notebook Conventions

### Naming

All notebooks follow the pattern `NN_descriptive_name.ipynb` where `NN` is a two-digit number indicating logical order. This makes the exploration narrative clear to anyone browsing the project and establishes a natural reading order.

### Standard Header

Every notebook begins with a markdown cell containing the title, purpose (one to two sentences explaining what this notebook investigates), author, date, and a list of key findings or conclusions (filled in after the analysis is complete).

### Cell Organization

Notebooks should follow this cell structure: (1) header and purpose, (2) imports and configuration, (3) data loading, (4) analysis sections with markdown headers, (5) conclusions and next steps. Keep individual cells focused on a single operation. Avoid cells longer than 30 lines.

### Output Policy

Commit notebooks with outputs cleared to keep the repository lean and avoid merge conflicts on binary cell outputs. The `.pre-commit-config.yaml` includes a hook that strips outputs automatically. When specific outputs must be preserved for documentation, save them as standalone files in `outputs/figures/` and reference them from the notebook.

## Configuration Variables

| Variable | Description | Default |
|---|---|---|
| `{{project_name}}` | Human-readable project name | Required |
| `{{project_slug}}` | Directory name | Auto-generated |
| `{{package_name}}` | Python import name for utilities | Auto-generated |
| `{{author_name}}` | Researcher name | Required |
| `{{email}}` | Researcher email | Required |
| `{{description}}` | Research question or objective | Required |
| `{{python_version}}` | Python version | 3.11 |

## Dependencies

```toml
[dependencies]
python = ">=3.11"
jupyterlab = ">=4.0"
ipywidgets = ">=8.0"
matplotlib = ">=3.8"
seaborn = ">=0.13"
pandas = ">=2.1"
numpy = ">=1.26"
pytorch = ">=2.0"
torchvision = ">=0.16"
albumentations = ">=1.3"
pillow = ">=10.0"
scikit-learn = ">=1.3"
```

## Usage

### Getting Started

```bash
# Install dependencies
pixi install

# Download or prepare data
pixi run python scripts/setup_data.py

# Launch JupyterLab
pixi run jupyter lab
```

### Working with Notebooks

```bash
# Start with data exploration
# Open notebooks/01_data_exploration.ipynb in JupyterLab

# Create a new notebook from the template
cp notebooks/_template.ipynb notebooks/05_new_experiment.ipynb

# Run all notebooks non-interactively for validation
pixi run jupyter nbconvert --execute notebooks/01_data_exploration.ipynb
```

### Using Utility Modules

```python
# Inside any notebook
from {{package_name}}.config import PATHS
from {{package_name}}.plotting import plot_class_distribution, plot_sample_grid
from {{package_name}}.data_utils import load_dataset, create_splits

# Load data using centralized paths
dataset = load_dataset(PATHS.raw / "images", PATHS.raw / "labels.csv")

# Generate standard visualizations
plot_class_distribution(dataset.labels, save_path=PATHS.figures / "class_dist.png")
plot_sample_grid(dataset, n=16, save_path=PATHS.figures / "samples.png")
```

### Exporting to Scripts

```bash
# Convert a notebook to a standalone Python script
pixi run python scripts/export_notebook.py notebooks/03_baseline_model.ipynb scripts/baseline.py

# The exported script preserves code cells and converts markdown to comments
```

## Integration with Experiment Tracking

The archetype includes an optional experiment tracking notebook (`04_experiment_tracking.ipynb`) that demonstrates integration with common tracking platforms. To enable tracking in other notebooks, use the utility functions provided.

```python
# Lightweight tracking with local JSON logs
from {{package_name}}.eval_utils import log_experiment

results = {"accuracy": 0.94, "f1": 0.91, "model": "resnet50", "epochs": 10}
log_experiment("experiment_001", results, save_dir=PATHS.reports)
```

For production-grade tracking, the archetype integrates with Weights and Biases, MLflow, or TensorBoard through optional dependencies that can be added to `pixi.toml`.

## Customization Guide

### Adding New Utility Modules

Place reusable functions in `src/{{package_name}}/` rather than duplicating them across notebooks. Common additions include custom augmentation pipelines, domain-specific evaluation metrics, and specialized data parsers. Every utility module should have corresponding tests in `tests/`.

### Graduating to a Training Project

When an experiment proves successful and needs to scale, use the PyTorch Training Project archetype to create a production training codebase. Copy the validated model architecture from the notebook utility modules, translate Hydra configs from the notebook parameters, and set up proper data loading with the LightningDataModule pattern.

### Managing Large Data

For datasets too large for the repository, keep the data in object storage and commit only a lightweight, versioned manifest that records each file's path and content hash. Update `scripts/setup_data.py` to read that manifest and fetch the referenced files into `data/`.

### Custom Plotting Styles

Edit `src/{{package_name}}/plotting.py` to define a consistent visual style across all notebooks. Set matplotlib rcParams, define a project color palette, and create template functions for common plot types (confusion matrices, ROC curves, training loss curves) that produce publication-ready figures.

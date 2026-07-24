# Research Notebook

Jupyter-based experimentation environment with reproducibility, visualization, and experiment tracking integration.

## Purpose

This archetype provides a structured notebook environment for ML research and experimentation. It enforces notebook conventions (imports at top, markdown documentation, clean outputs), integrates with experiment tracking, and includes utility modules for common research tasks like visualization and data exploration.

## Directory Structure

```
{{project_slug}}/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── utils.py               # Shared notebook utilities
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── models/
│   ├── data/
│   └── visualization/
│       ├── __init__.py
│       └── plots.py           # Reusable plotting functions
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── figures/
│   └── results/
├── tests/
│   └── test_utils.py
└── ...
```

## Notebook Conventions

1. **Numbered prefixes** -- `01_`, `02_` for execution order
2. **Markdown cells** -- document purpose, methodology, and findings
3. **Imports at top** -- all imports in the first code cell
4. **Clean outputs** -- clear outputs before committing (pre-commit hook)
5. **Reusable code in modules** -- move repeated code to `src/`

## Usage

```bash
# Start Jupyter
jupyter lab

# Convert notebook to script
jupyter nbconvert --to script notebooks/01_data_exploration.ipynb

# Run all notebooks headless (for CI)
pytest --nbmake notebooks/
```

## Customization

- Add new notebooks following the numbered convention
- Move reusable visualization code to `src/{{package_name}}/visualization/`
- Configure experiment tracking (W&B, MLflow) for notebook experiments
- Add data processing utilities to `notebooks/utils.py`

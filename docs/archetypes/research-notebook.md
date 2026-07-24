# Research Notebook

Jupyter-based experimentation environment with reproducibility, visualization, and experiment tracking integration.

## Purpose

This archetype provides a structured notebook environment for ML research and experimentation. It enforces notebook conventions (imports at top, markdown documentation, clean outputs), integrates with experiment tracking, and includes utility modules for common research tasks like visualization and data exploration.

## Directory Structure

```
${project_slug}/
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── notebooks/
│   ├── 01-abc-explore-dataset.ipynb
│   ├── README.md
│   └── _template.ipynb
├── outputs/
│   ├── figures/
│   │   └── .gitkeep
│   └── reports/
│       └── .gitkeep
├── src/
│   └── ${package_name}/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── py.typed
│       └── viz.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data.py
│   └── test_viz.py
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── pixi.toml
└── pyproject.toml
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

# Research Notebook Archetype

A structured Jupyter-based project template for rapid experimentation, prototyping, and research exploration in computer vision and machine learning. This archetype provides an organized framework for exploratory data analysis, model prototyping, paper reproduction, and experiment documentation while maintaining enough structure to graduate promising experiments into production code.

## Purpose

Research work in machine learning is inherently iterative and exploratory. Notebooks are the natural medium for this work because they combine code execution, visualization, and narrative documentation in a single artifact. However, unstructured notebook projects quickly devolve into a tangle of unnamed files, hidden state dependencies, and unreproducible results.

The Research Notebook archetype solves this by imposing two lightweight rules and shipping the tooling that enforces them:

1. **Notebooks explore; `src/` remembers.** A cell worth running twice becomes a typed, tested function in `src/<package>/` and is imported back into the notebook. Notebooks stay narrative — question, look, plot, conclude.
2. **Notebooks stay diff-able.** Committed notebooks carry no outputs and no execution counts, enforced by an `nbstripout` pre-commit hook. A notebook then reviews like source code rather than like a binary blob.

Everything else follows from those two. Data paths come from a single `PATHS` object, every run is described by one validated Pydantic `ExperimentConfig`, and randomness goes through a seeded generator — so a notebook's results survive being cleared and rerun.

## Use Cases

- **Exploratory data analysis (EDA)** -- Investigate dataset statistics, class distributions, annotation quality, and outliers before committing to a training approach.
- **Model exploration and prototyping** -- Quickly test model architectures, loss functions, and training strategies with small data subsets before scaling up.
- **Paper reproduction** -- Reproduce published results in a documented, shareable format with clear methodology notes.
- **Visualization and reporting** -- Generate publication-quality figures, confusion matrices, and performance plots for stakeholder communication.
- **Ablation studies** -- Systematically vary components of a pipeline and document the impact of each change.
- **Dataset curation** -- Interactively inspect, filter, relabel, and augment training data with visual feedback.

## Directory Structure

This is exactly what `whet init research-notebook` writes — no more, no less.

```
${project_slug}/
├── .gitignore                          # ignores checkpoints, data/, outputs/
├── .pre-commit-config.yaml             # nbstripout + ruff + mypy + hygiene hooks
├── pixi.toml                           # environment and tasks (lab, test, quality)
├── pyproject.toml                      # deps, ruff, mypy strict, pytest
├── README.md
├── notebooks/
│   ├── README.md                       # naming convention + reproducibility checklist
│   ├── _template.ipynb                 # copy this to start a new notebook
│   └── 01-abc-explore-dataset.ipynb    # runnable worked example
├── src/${package_name}/
│   ├── __init__.py
│   ├── py.typed
│   ├── config.py                       # Pydantic V2 ExperimentConfig + PATHS
│   ├── data.py                         # seeded RNG, synthetic data, splits, summaries
│   └── viz.py                          # reusable Matplotlib helpers + project style
├── data/
│   ├── raw/.gitkeep                    # original, immutable inputs
│   └── processed/.gitkeep              # derived data from a reproducible step
├── outputs/
│   ├── figures/.gitkeep                # saved figures (not cell outputs)
│   └── reports/.gitkeep                # metric dumps and exported tables
└── tests/
    ├── __init__.py
    ├── conftest.py                     # headless Agg backend, shared fixtures
    ├── test_config.py
    ├── test_data.py
    └── test_viz.py
```

The generated project has **no torch and no GPU dependency**. Its runtime deps are
`numpy`, `matplotlib`, `pydantic`, and `loguru`; Jupyter lives in a `notebook`
extra. The example notebook runs end to end on a fresh clone with no download.

## Key Features

- **Runnable example notebook** (`01-abc-explore-dataset.ipynb`) that imports every helper from `src/`, demonstrating the discipline rather than describing it.
- **`nbstripout` pre-commit hook** so cell outputs and execution counts never reach git.
- **Numbered, initialled notebook convention** (`NN-initials-topic.ipynb`) for clear ownership and reading order.
- **Notebook template** (`_template.ipynb`) with the standard header, setup, and conclusions cells.
- **Importable, tested utility modules** under `src/` with `py.typed`, so notebooks stay short.
- **Pydantic V2 `ExperimentConfig`** — frozen, `extra="forbid"`, with a `variant()` helper for ablations and validation that fires at construction time.
- **Centralized `PATHS`** for `data/raw`, `data/processed`, `outputs/figures`, `outputs/reports`.
- **Loguru everywhere** — no `print`, no stdlib `logging`.
- **Full quality gate**: `ruff` (line-length 100; `E,F,I,N,UP,S,B,A,C4,T20,SIM`), `mypy --strict`, and `pytest`.

## Notebook Conventions

### Naming

Notebooks follow `NN-initials-topic.ipynb` — for example `01-abc-explore-dataset.ipynb`. `NN` gives a reading order, the initials say who owns the file so parallel work does not collide, and the topic is lowercase and hyphenated. Numbers are a reading order, not a dependency chain: a notebook that only runs after another has been executed is a bug, so load from `data/processed/` instead.

### Standard Header

Every notebook begins with a markdown cell containing the title, the **question** it is trying to answer, the author, the status, and a **findings** section filled in once the analysis is complete.

### Cell Organization

Notebooks follow this structure: (1) header and question, (2) setup — imports and project style, (3) configuration — one validated `ExperimentConfig`, (4) load via a helper in `src/`, (5) analysis sections under markdown headers, (6) conclusions and next steps. Keep cells under roughly 30 lines; a long cell is usually a function that has not been moved to `src/` yet.

### Output Policy

Commit notebooks with outputs cleared. The `nbstripout` hook in `.pre-commit-config.yaml` does this automatically on every commit. Outputs worth preserving are saved as real files under `outputs/figures/` via `viz.save_figure(fig, cfg.figure_path("overview"))` and referenced from the notebook.

### Importing from `src/`

`whet init` substitutes `${...}` variables in text files, but `.ipynb` files are copied byte-for-byte by the scaffold engine — so the shipped notebooks cannot hard-code the generated package name. Their setup cell resolves it from the source tree instead:

```python
PACKAGE = next(p.name for p in sorted(SRC.iterdir()) if (p / "__init__.py").is_file())
config_mod = import_module(f"{PACKAGE}.config")
```

In notebooks you write yourself, replace that with a plain `from your_package import ...`.

## Configuration Variables

These are the only variables the scaffold engine substitutes.

| Variable | Description | Default |
|---|---|---|
| `${project_name}` | Human-readable project name | Required |
| `${project_slug}` | Directory and distribution name | Derived from `project_name` |
| `${package_name}` | Python import name for the `src/` package | Derived from `project_slug` |
| `${description}` | Research question or objective | Empty |
| `${author}` | Researcher name | Empty |
| `${python_version}` | Minimum Python version | 3.11 |

## Dependencies

Deliberately light — the archetype must install and test without a GPU.

```toml
# pyproject.toml [project].dependencies
numpy      = ">=1.26"
matplotlib = ">=3.8"
pydantic   = ">=2.6"
loguru     = ">=0.7"

# [project.optional-dependencies].notebook
jupyterlab = ">=4.0"
ipykernel  = ">=6.29"
ipywidgets = ">=8.1"
nbstripout = ">=0.7"

# [project.optional-dependencies].dev
pytest, pytest-cov, ruff, mypy, pre-commit
```

Add anything else with pixi:

```bash
pixi add scikit-learn pandas seaborn   # conda-forge
pixi add --pypi some-package           # PyPI only
```

## Usage

### Getting Started

```bash
pixi install
pixi run hooks        # install pre-commit, including nbstripout
pixi run lab          # launch JupyterLab in notebooks/
```

### Working with Notebooks

```bash
# Start from the worked example
#   notebooks/01-abc-explore-dataset.ipynb

# Create a new notebook from the template
cp notebooks/_template.ipynb notebooks/02-abc-augmentation-sweep.ipynb

# Validate a notebook non-interactively (proves it still runs top to bottom)
pixi run nbrun

# Strip outputs manually if you are not using the hooks
pixi run nbstrip
```

### Quality Gate

Tool-agnostic commands, runnable in any environment that has the deps:

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Or via pixi, which supplies the environment:

```bash
pixi run quality      # lint + format-check + typecheck + test
```

### Using the Utility Modules

```python
# Inside any notebook, after the setup cell
cfg = config_mod.ExperimentConfig(name="explore-dataset", seed=42, n_samples=600)
cfg.log_summary()

features, labels = data_mod.make_synthetic_dataset(
    n_samples=cfg.n_samples, n_classes=cfg.n_classes, seed=cfg.seed
)
train_idx, val_idx, test_idx = data_mod.split_indices(
    cfg.n_samples, cfg.train_fraction, cfg.val_fraction, seed=cfg.seed
)

viz_mod.plot_class_distribution(labels)
viz_mod.save_figure(fig, cfg.figure_path("overview"))   # -> outputs/figures/
```

Ablations derive from the base config rather than mutating it:

```python
ablation = cfg.variant(seed=7, stage="ablation")   # validated copy
```

## Customization Guide

### Adding New Utility Modules

Place reusable functions in `src/${package_name}/` rather than duplicating them across notebooks. Common additions include dataset loaders, custom augmentation pipelines, domain-specific evaluation metrics, and specialized parsers. Every utility module gets corresponding tests in `tests/` — if a helper is too tangled to test, it is too tangled to trust inside a notebook.

### Replacing the Synthetic Data

`data.make_synthetic_dataset` exists so the example notebook runs on a fresh clone with no download. Replace it with a real loader that reads from `PATHS.raw`, keep the same return signature, and the notebook and its tests keep working.

### Graduating to a Training Project

When an experiment proves successful and needs to scale, use the PyTorch Training Project archetype to create a production training codebase. Move the validated helpers out of `src/${package_name}/`, translate `ExperimentConfig` fields into Hydra configs, and set up proper data loading with the LightningDataModule pattern. Leave the notebook behind as the record of *why*.

### Managing Large Data

`data/` and `outputs/` are git-ignored except for their `.gitkeep` markers. For datasets too large for the repository, keep the data in object storage and commit a lightweight, versioned manifest recording each file's path and content hash; fetch from that manifest into `data/raw/`.

### Custom Plotting Styles

Edit `FIGURE_STYLE` in `src/${package_name}/viz.py` to define a consistent visual identity across the project, and call `viz.use_project_style()` once in each notebook's setup cell. Add template functions for common plot types (confusion matrices, ROC curves, training loss curves) alongside the existing helpers so every figure in the project looks the same without per-plot fiddling.

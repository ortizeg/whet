# ${project_name}

${description}

A research project where **notebooks are first-class but code is not**. Notebooks
under `notebooks/` carry the narrative — the question, the look, the plot, the
conclusion. Everything reusable lives in `src/${package_name}/`, typed and tested,
and gets imported back into the notebook.

## Layout

```
.
├── notebooks/
│   ├── README.md                       # naming convention and conventions
│   ├── _template.ipynb                 # copy this to start a new notebook
│   └── 01-abc-explore-dataset.ipynb    # worked example
├── src/${package_name}/
│   ├── __init__.py
│   ├── py.typed
│   ├── config.py                       # Pydantic V2 experiment config + PATHS
│   ├── data.py                         # loading, splitting, summarising
│   └── viz.py                          # reusable Matplotlib helpers
├── tests/                              # real tests for everything in src/
├── data/{raw,processed}/               # git-ignored, .gitkeep tracked
├── outputs/{figures,reports}/          # git-ignored, .gitkeep tracked
├── .pre-commit-config.yaml             # nbstripout + ruff + mypy
├── pixi.toml
└── pyproject.toml
```

## Setup

```bash
pixi install
pixi run hooks        # installs pre-commit, including nbstripout
```

Add a dependency:

```bash
pixi add scikit-learn          # conda-forge
pixi add --pypi some-package   # PyPI only
```

## Working

```bash
jupyter lab           # explore
pytest                # test the code your notebooks import
ruff check .          # lint
ruff format .         # format
mypy src/ --strict    # type-check
```

Or through pixi, which supplies the environment:

```bash
pixi run lab
pixi run quality      # lint + format-check + typecheck + test
```

## The discipline

**1. Notebooks explore; `src/` remembers.**
A cell worth running twice is a function. Move it to `src/${package_name}/`, give
it a signature and a test, and import it back:

```python
counts = data_mod.class_counts(labels)
viz_mod.plot_class_distribution(labels)
```

**2. Notebooks stay diff-able.**
Committed notebooks carry no outputs and no execution counts. The `nbstripout`
pre-commit hook enforces this, so a notebook reviews like source instead of like
a binary blob. Figures worth keeping are written to `outputs/figures/`:

```python
viz_mod.save_figure(fig, cfg.figure_path("overview"))
```

**3. No magic numbers, no literal paths.**
Every run is described by one validated `ExperimentConfig`, and every directory
comes from `PATHS`:

```python
cfg = config_mod.ExperimentConfig(name="explore-dataset", seed=42, n_samples=600)
cfg.log_summary()
ablation = cfg.variant(seed=7, stage="ablation")
```

Invalid values fail at construction time, not three cells later.

**4. Reproducible or it did not happen.**
Randomness goes through `data_mod.rng_from_seed(cfg.seed)` — never a global
`np.random.seed`. Before committing, **Kernel > Restart & Run All** must pass;
`data/` and `outputs/` are git-ignored because they are regenerable.

**5. Loguru, never `print`.**

```python
from loguru import logger

logger.info("Loaded {} samples across {} classes", features.shape[0], cfg.n_classes)
```

See `notebooks/README.md` for the naming convention and the full reproducibility
checklist.

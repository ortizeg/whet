# Notebooks

Notebooks are for **exploration**. Reusable code lives in `src/${package_name}/`.

That single rule is what this archetype exists to enforce. A notebook is a
narrative — load, look, plot, conclude. The moment a cell is worth running twice,
it stops being narrative and becomes code: move it into `src/`, give it a type
signature, and write a test for it. The notebook then imports it back.

## Naming convention

```
NN-initials-topic.ipynb
```

| Part | Meaning | Example |
|---|---|---|
| `NN` | Two-digit sequence number giving a reading order | `01`, `02`, `10` |
| `initials` | Who owns the notebook, so parallel work does not collide | `abc`, `egs` |
| `topic` | Lowercase, hyphenated, specific | `explore-dataset` |

Examples:

```
01-abc-explore-dataset.ipynb
02-abc-augmentation-sweep.ipynb
03-jkl-baseline-ablation.ipynb
```

`_template.ipynb` is the starting point — copy it, do not edit it in place:

```bash
cp notebooks/_template.ipynb notebooks/02-abc-augmentation-sweep.ipynb
```

Numbers are a reading order, not a dependency chain. A notebook that only runs
after another notebook has been executed is a bug: make it self-contained by
loading from `data/processed/` instead.

## Standard structure

Every notebook opens with a markdown cell stating the **question**, the author,
the status, and a **findings** section filled in once the analysis is done. Then:

1. Setup — imports and style, one cell
2. Configuration — one validated `ExperimentConfig`, no loose magic numbers
3. Load — through a helper in `src/`, never an inline path string
4. Analysis — short cells under markdown headers
5. Conclusions and next steps

Keep cells under roughly 30 lines. A long cell is usually a function that has not
been moved to `src/` yet.

## Keeping notebooks diff-able

Committed notebooks carry **no outputs and no execution counts**. Outputs are
large, binary-ish, and change on every run, which turns any two-person project
into a stream of merge conflicts.

This is enforced by the `nbstripout` hook in `.pre-commit-config.yaml`:

```bash
pre-commit install          # once per clone
pre-commit run --all-files  # strip everything now
```

If you are not using the hooks, strip manually before committing:

```bash
nbstripout notebooks/*.ipynb
```

Anything worth preserving is saved as a real file:

```python
viz_mod.save_figure(fig, cfg.figure_path("overview"))   # -> outputs/figures/
```

`outputs/` and `data/` are git-ignored — results are reproducible from code plus
a seed, and both are pinned by `ExperimentConfig`.

## Reproducibility checklist

Before you commit or share a notebook:

- **Kernel > Restart & Run All** completes without error — hidden state from
  out-of-order execution is the most common source of unreproducible results.
- Every random draw goes through a seeded generator (`data_mod.rng_from_seed`),
  never a global `np.random.seed`.
- No filesystem path is typed as a literal; all of them come from `PATHS`.
- The findings section at the top is filled in.

Validate a notebook non-interactively at any time:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01-abc-explore-dataset.ipynb
```

## Promoting a notebook

When an experiment settles, promote it: move the functions into
`src/${package_name}/`, add tests under `tests/`, and leave the notebook as the
record of *why* — the question it answered and what it found.

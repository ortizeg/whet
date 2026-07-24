# ${project_name}

${description}

A reproducible ETL pipeline for CV/ML datasets: **ingest → validate → split → write**.
Bulk data stays out of Git; a content-hashed manifest and a small dataset card describe
each version.

## Why this exists

The highest-risk step in any CV pipeline is the train/val/test split. Frames from the same
clip, images of the same patient, and plays from the same match are near-duplicates — if
they straddle splits, validation measures memorization and the model fails silently in
production. This project **splits on the group, never on the row**, and asserts split
disjointness inside the pipeline (not only in tests), so a leak aborts the run.

## Layout

```
${project_slug}/
├── conf/pipeline.toml            # validated pipeline configuration
├── pixi.toml                     # environment + tasks
├── pyproject.toml                # packaging, ruff, mypy, pytest
├── src/${package_name}/
│   ├── config.py                 # frozen Pydantic V2 config models
│   ├── manifest.py               # ImageRecord, content hashing, Parquet manifest
│   ├── splitting.py              # group-aware, leakage-free splitting
│   ├── quality.py                # data-quality gate (errors abort the run)
│   ├── stages.py                 # ingest / validate / split / write
│   ├── pipeline.py               # orchestration + PipelineResult
│   ├── sample_data.py            # synthetic dataset generator (stdlib only)
│   └── cli.py                    # argparse entry point
└── tests/                        # incl. the split-disjointness tests
```

`data/` is created on demand and is Git-ignored; only `data/processed/dataset.json`
(the fingerprinted dataset card) is meant to be committed.

## Setup

```bash
pixi install
```

Adding a dependency:

```bash
pixi add polars
pixi add --feature dev pytest
```

## Expected raw layout

```
data/raw/<label>/<group_id>/<file>.png
data/raw/goal/match_0007/frame_000123.png
```

The middle directory is the **grouping key** — one clip, one patient, one session. Change
`ingest.label_depth` / `ingest.group_depth` in `conf/pipeline.toml` if your tree differs.

## Usage

```bash
# 1. Generate a small synthetic dataset to try things out (optional)
python -m ${package_name} sample --root data/raw

# 2. Scan the raw tree and report the manifest
python -m ${package_name} scan --config conf/pipeline.toml

# 3. Run the data-quality gate only
python -m ${package_name} validate --config conf/pipeline.toml

# 4. Run the whole pipeline
python -m ${package_name} run --config conf/pipeline.toml
```

Outputs land in `data/processed/`:

| Artifact | Contents |
|---|---|
| `manifest.parquet` | one content-hashed row per sample |
| `splits/train.parquet`, `splits/val.parquet`, `splits/test.parquet` | per-split manifests |
| `dataset.json` | fingerprint, record/group counts, label list, split sizes |

The same commands are wired as pixi tasks: `pixi run sample`, `pixi run scan`,
`pixi run validate`, `pixi run run`.

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Or via pixi: `pixi run quality`.

## Extending the pipeline

- **New stage** — add a function to `stages.py` with explicit inputs/outputs and call it
  from `run_pipeline`. There is deliberately no abstract stage base class.
- **New quality check** — add a `_check_*` function in `quality.py` returning
  `QualityIssue`s and call it from `validate_manifest`. `ERROR` aborts, `WARNING` logs.
- **New manifest field** — add it to `ImageRecord` *and* `MANIFEST_SCHEMA`, bump
  `SCHEMA_VERSION`, and write a migration. Never mutate a schema silently.
- **New source format** — write a reader that yields `ImageRecord`s and swap it into
  `stage_ingest`; everything downstream is format-agnostic.

## Conventions

- **Loguru** for all logging — never `print()`.
- **Pydantic V2** frozen models for every config and record.
- **Parquet** (zstd) for all tabular data — never CSV.
- **src-layout** with a `py.typed` marker; `mypy --strict` clean.
- Splits are deterministic for a given seed: group order comes from a seeded SHA-256
  ranking, so results reproduce across machines and library versions.

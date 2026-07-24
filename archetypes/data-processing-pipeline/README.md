# Data Processing Pipeline Archetype

A working project template for ETL workflows that turn raw CV/ML data into validated,
leakage-free, versioned training splits. Generated projects run end to end on day one:
`ingest -> validate -> split -> write`, with a content-hashed Parquet manifest, a data-quality
gate, and a group-aware splitter that aborts the run if a group straddles two splits.

## Purpose

Data preparation is the most time-consuming and error-prone phase of any machine learning
project. Raw datasets arrive in inconsistent formats, contain corrupt files, have labeling
errors, and require extensive transformation before they are suitable for training. Despite
this, data processing code is often the least structured part of an ML codebase -- scattered
across ad-hoc scripts, undocumented Jupyter cells, and bash one-liners that are impossible to
reproduce.

This archetype gives that work a spine: each processing step is an ordinary, testable function
with explicit inputs and outputs; every sample is content-hashed; every configuration value is
validated by Pydantic at load time; and the highest-risk step -- splitting -- is protected by an
assertion that runs inside the pipeline, not only in the test suite.

The core design principle is that every transformation applied to data must be explicit, tested,
logged, and reproducible. When model performance changes, the data lineage traces back to a
dataset fingerprint, so you can tell whether the change came from the data or the model.

## Use Cases

- **Train/val/test splitting** -- Reproducible splits that keep every frame of a clip, every
  image of a patient, and every play of a match inside a single split.
- **Dataset manifesting** -- Turn a directory tree into a validated, content-hashed Parquet
  manifest with a stable fingerprint per dataset version.
- **Quality assurance** -- Automated checks for empty datasets, corrupt/undersized files,
  duplicate content, thin labels, mixed-label groups, and post-split label coverage.
- **Dataset versioning** -- Bulk data stays in object storage; a small `dataset.json` card with
  the fingerprint, counts, and split sizes is what gets committed.
- **Dataset cleaning** -- Extend the ingest stage with format normalization and corrupt-file
  removal; the quality gate already fails the run loudly.
- **Annotation format conversion** -- Swap the reader in `stage_ingest`; everything downstream is
  format-agnostic because it only sees `ImageRecord`s.

## Directory Structure

The generated project (`whet init data-processing-pipeline`):

```
${project_slug}/
├── .gitignore
├── README.md
├── pixi.toml                            # environment + tasks
├── pyproject.toml                       # packaging, ruff, mypy, pytest config
├── conf/
│   └── pipeline.toml                    # validated pipeline configuration
├── src/${package_name}/
│   ├── __init__.py                      # public API re-exports
│   ├── py.typed
│   ├── config.py                        # frozen Pydantic V2 config models
│   ├── manifest.py                      # ImageRecord, hashing, Parquet manifest, dataset card
│   ├── splitting.py                     # group-aware, leakage-free splitting
│   ├── quality.py                       # data-quality gate (ERROR aborts the run)
│   ├── stages.py                        # ingest / validate / split / write
│   ├── pipeline.py                      # orchestration + PipelineResult
│   ├── sample_data.py                   # stdlib-only synthetic dataset generator
│   ├── cli.py                           # argparse entry point
│   └── __main__.py                      # python -m ${package_name}
└── tests/
    ├── __init__.py
    ├── conftest.py                      # synthetic manifests + synthetic raw tree
    ├── test_splitting.py                # split disjointness, determinism, leak detection
    ├── test_manifest.py                 # schema, hashing, Parquet round-trip
    ├── test_quality.py                  # every quality check
    └── test_pipeline.py                 # end-to-end pipeline and CLI
```

`data/` is created on demand at runtime and is Git-ignored, except
`data/processed/dataset.json`.

## Key Features

- **Group-aware splitting** -- splits on the grouping key (clip, patient, match, session), never
  on the row, and calls `assert_no_group_leakage` inside the pipeline so a leak aborts the run.
- **Deterministic without an RNG** -- group order comes from a seeded SHA-256 ranking, so splits
  reproduce across machines, Python versions, and library upgrades.
- **Content hashing** -- SHA-256 per sample for deduplication and corruption detection, plus an
  order-independent dataset fingerprint.
- **Pydantic V2 validation** -- frozen config models; ratios that do not sum to 1.0 fail at load
  time, not three hours in.
- **Quality gate with severities** -- `ERROR` aborts, `WARNING` logs; checks run after ingest
  *and* after splitting, because a healthy manifest still yields bad splits with bad ratios.
- **Parquet everywhere** -- zstd-compressed manifest and per-split manifests; never CSV.
- **Loguru structured logging** -- realized per-split record/group/label counts at every stage
  boundary, so skewed ratios are visible immediately.
- **Runnable on day one** -- a stdlib-only generator writes a synthetic clip-structured dataset
  of real PNGs, so `sample -> run` works before you have any data.

## Pipeline Stages

Stages are plain functions, not a class hierarchy -- add one by writing a function and calling it
from `run_pipeline`.

```python
def stage_ingest(config: IngestConfig) -> pl.DataFrame: ...
def stage_validate(manifest: pl.DataFrame, config: QualityConfig) -> QualityReport: ...
def stage_split(manifest: pl.DataFrame, config: SplitConfig) -> dict[str, pl.DataFrame]: ...
def stage_write(
    manifest: pl.DataFrame,
    splits: Mapping[str, pl.DataFrame],
    config: OutputConfig,
    quality: QualityConfig,
) -> dict[str, Path]: ...
```

The leakage guard, which runs in the pipeline and not only in tests:

```python
def assert_no_group_leakage(splits: Mapping[str, pl.DataFrame], group_column: str) -> None:
    """Raise DataLeakageError if any group appears in more than one split."""
```

## Expected Raw Layout

```
data/raw/<label>/<group_id>/<file>.png
data/raw/goal/match_0007/frame_000123.png
```

The middle directory is the grouping key. Adjust `ingest.label_depth` / `ingest.group_depth` in
`conf/pipeline.toml` for a different tree.

## Template Variables

| Variable | Description | Default |
|---|---|---|
| `${project_name}` | Human-readable project name | Required |
| `${project_slug}` | Directory / distribution name | Derived from project name |
| `${package_name}` | Python import name | Derived from slug |
| `${description}` | Pipeline purpose description | Empty |
| `${author}` | Author name | Empty |
| `${python_version}` | Minimum Python version | 3.11 |

## Dependencies

Deliberately light -- no torch, no GPU, no image-decoding dependency in the core path.

```toml
[dependencies]
python = ">=3.11"
pydantic = ">=2.6"
loguru = ">=0.7"
polars = ">=1.0"

[feature.dev.dependencies]
pytest = ">=7.4"
pytest-cov = ">=4.1"
ruff = ">=0.8"
mypy = ">=1.11"
```

## Usage

```bash
pixi install

# Generate a synthetic dataset to try the pipeline before you have data (optional)
python -m ${package_name} sample --root data/raw

# Scan the raw tree and report the manifest fingerprint
python -m ${package_name} scan --config conf/pipeline.toml

# Run the quality gate only (non-zero exit on failure)
python -m ${package_name} validate --config conf/pipeline.toml

# Run the full pipeline
python -m ${package_name} run --config conf/pipeline.toml
```

Equivalent pixi tasks: `pixi run sample`, `pixi run scan`, `pixi run validate`, `pixi run run`.

Outputs in `data/processed/`: `manifest.parquet`, `splits/{train,val,test}.parquet`, and
`dataset.json` (fingerprint, counts, labels, split sizes).

### Development

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Or `pixi run quality` for all four.

## Customization Guide

### Adding a Pipeline Stage

1. Write a function in `src/${package_name}/stages.py` with explicit inputs and outputs.
2. Add its configuration as a frozen Pydantic model in `config.py` and hang it off
   `PipelineConfig`.
3. Call it from `run_pipeline` in `pipeline.py`.
4. Add a test in `tests/`. There is no stage registry or abstract base class to update.

### Adding a Quality Check

1. Write a `_check_*` function in `quality.py` returning a list of `QualityIssue`.
2. Call it from `validate_manifest` (or `validate_splits` for post-split checks).
3. Use `Severity.ERROR` to abort the run and `Severity.WARNING` to log and continue.

### Adding a Source Format

1. Write a reader that yields `ImageRecord`s (COCO, VOC, YOLO, a CSV of URLs, a database query).
2. Swap it into `stage_ingest`. Splitting, validation, and writing are format-agnostic.
3. If the reader needs new fields, add them to `ImageRecord` *and* `MANIFEST_SCHEMA`, bump
   `SCHEMA_VERSION`, and write an explicit migration -- never mutate a schema silently.

### Choosing the Grouping Key

`split.group_column` defaults to `group_id`. Point it at whatever makes samples correlated:
video/clip ID, patient/subject ID, match/session ID, player identity, camera or site ID, or
capture date for time-correlated data. Set it to `null` only when samples are genuinely
independent -- the pipeline then falls back to a stratified split and logs a warning.

### Scaling Up

Ingest is a single-process directory walk, which is fine to tens of thousands of files. Beyond
that, hash files in a `ProcessPoolExecutor` inside `stage_ingest`, and switch the manifest reads
to `pl.scan_parquet` for lazy, out-of-core execution. The manifest, splitting, and quality code
paths do not change.

## Not Included

Deliberately out of scope, to keep the generated project small and dependency-light -- add them
as your project needs them: augmentation policies (Albumentations), HTML dataset reports,
notebooks, DVC integration, and CI workflow files.

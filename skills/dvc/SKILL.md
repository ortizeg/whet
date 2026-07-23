---
name: dvc
description: >
  Use this skill when versioning large datasets, model weights, or pipeline artifacts
  with DVC — running dvc init/add/push/pull, configuring S3/GCS/Azure remotes, defining
  reproducible dvc.yaml pipelines, and keeping big binaries out of Git. Reach for it any
  time data or checkpoints are too large to commit to Git and need tracking or sharing,
  even if the user doesn't say "DVC". For experiment metrics, run comparison, and model
  registry see mlflow or wandb.
---

# Data Version Control (DVC) for ML Projects

## Overview

DVC (Data Version Control) is an open-source version control system for machine learning projects. It extends Git to handle large files, datasets, and ML models that cannot be stored in Git repositories. DVC tracks these files using lightweight metafiles (`.dvc` files) that are committed to Git, while the actual data is stored in configurable remote storage (S3, GCS, Azure, or local). DVC is opt-in and should be used when projects involve datasets or model files too large for Git.

## Why Use DVC

Machine learning projects face a unique challenge: the code is small and fits in Git, but the data and models are large and do not. Without DVC, teams resort to ad-hoc solutions like shared network drives, manual file naming conventions (`model_v3_final_FINAL.pt`), or storing everything in cloud buckets without any version history. DVC solves these problems:

- **Version control for data and models** with the same Git workflow developers already know.
- **Reproducible pipelines** that connect data, code, and outputs.
- **Remote storage** on any cloud provider or local filesystem.
- **Lightweight tracking** using small `.dvc` metafiles committed to Git.
- **Experiment tracking** for comparing parameter and metric changes.
- **Collaboration** through shared remote storage.
- **No vendor lock-in** with support for S3, GCS, Azure, SSH, and local storage.

## Installation and Initialization

### Installation

```bash
# Using pip
pip install dvc

# With specific remote storage support
pip install dvc[s3]    # Amazon S3
pip install dvc[gs]    # Google Cloud Storage
pip install dvc[azure] # Azure Blob Storage
pip install dvc[ssh]   # SSH/SFTP

# Using pixi
pixi add dvc --feature data
pixi add dvc-s3 --feature data  # For S3 support
```

### Initialization

```bash
# Initialize DVC in an existing Git repository
cd my-project
dvc init

# This creates:
# .dvc/           - DVC internal directory
# .dvc/config     - DVC configuration
# .dvcignore      - Patterns to ignore (like .gitignore)

# Commit the DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

## Tracking Data Files

### Adding Files to DVC

```bash
# Track a dataset directory
dvc add data/coco/

# Track a model file
dvc add models/yolov8_best.pt

# Track a large file
dvc add data/video_dataset.tar.gz
```

Each `dvc add` command creates a `.dvc` metafile:

```yaml
# data/coco.dvc (auto-generated)
outs:
  - md5: d41d8cd98f00b204e9800998ecf8427e.dir
    size: 25432198
    nfiles: 118287
    hash: md5
    path: coco
```

### Git Integration

```bash
# The .dvc file is small and goes into Git
git add data/coco.dvc data/.gitignore
git commit -m "Track COCO dataset with DVC"

# The actual data is in .gitignore (auto-added by DVC)
# data/coco is now ignored by Git but tracked by DVC
```

### Checking Status

```bash
# See which tracked files have changed
dvc status

# See detailed diff
dvc diff
```

## Remote Storage

Remote storage is where DVC stores the actual data. Configure it once and all team members can push and pull data.

### Configuring Remote Storage

```bash
# Amazon S3
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote region us-east-1

# Google Cloud Storage
dvc remote add -d myremote gs://my-bucket/dvc-store

# Azure Blob Storage
dvc remote add -d myremote azure://my-container/dvc-store
dvc remote modify myremote account_name myaccount

# Local or network filesystem
dvc remote add -d myremote /mnt/shared/dvc-store

# SSH
dvc remote add -d myremote ssh://user@server:/path/to/dvc-store
```

The `-d` flag sets the remote as default. Configuration is stored in `.dvc/config`:

```ini
[core]
    remote = myremote
[remote "myremote"]
    url = s3://my-bucket/dvc-store
    region = us-east-1
```

### Pushing and Pulling Data

```bash
# Push all tracked data to remote
dvc push

# Pull all tracked data from remote
dvc pull

# Push/pull specific files
dvc push data/coco.dvc
dvc pull models/yolov8_best.pt.dvc

# Fetch data without checking out (download to cache only)
dvc fetch
```

## Pipeline Definitions

DVC pipelines define reproducible workflows that connect data, code, parameters, and outputs.

### Creating a Pipeline

Define stages in `dvc.yaml`:

```yaml
# dvc.yaml
stages:
  prepare:
    cmd: python src/prepare_data.py
    deps:
      - src/prepare_data.py
      - data/raw/
    params:
      - prepare.split_ratio
      - prepare.seed
    outs:
      - data/prepared/

  train:
    cmd: python src/train.py
    deps:
      - src/train.py
      - src/model.py
      - data/prepared/
    params:
      - train.epochs
      - train.learning_rate
      - train.batch_size
      - train.model_name
    outs:
      - models/best_model.pt
    metrics:
      - metrics/train_metrics.json:
          cache: false
    plots:
      - metrics/training_curves.csv:
          x: epoch
          y: loss

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/best_model.pt
      - data/prepared/test/
    metrics:
      - metrics/eval_metrics.json:
          cache: false
    plots:
      - metrics/confusion_matrix.csv:
          x: predicted
          y: actual
          template: confusion
```

### Parameters File

DVC reads parameters from `params.yaml` by default:

```yaml
# params.yaml
prepare:
  split_ratio: 0.8
  seed: 42

train:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  model_name: yolov8
```

### Running Pipelines

```bash
# Run the entire pipeline
dvc repro

# Run a specific stage
dvc repro train

# Force re-run even if nothing changed
dvc repro --force

# Run with modified parameters
# Edit params.yaml first, then:
dvc repro

# Visualize the pipeline DAG
dvc dag
```

Output of `dvc dag`:

```
+---------+
| prepare |
+---------+
      *
      *
      *
  +-------+
  | train |
  +-------+
      *
      *
      *
 +----------+
 | evaluate |
 +----------+
```

## Data and Model Versioning

DVC enables versioning by leveraging Git branches and tags.

### Versioning Workflow

```bash
# Version 1: Baseline model
dvc repro
git add dvc.lock params.yaml metrics/
git commit -m "v1: Baseline YOLOv8 model"
git tag v1.0

# Version 2: Updated augmentation
# Edit code and params
dvc repro
git add dvc.lock params.yaml metrics/
git commit -m "v2: Add mosaic augmentation"
git tag v2.0

# Switch between versions
git checkout v1.0
dvc checkout  # Restores data/model files to v1 state

git checkout v2.0
dvc checkout  # Restores data/model files to v2 state
```

### Comparing Versions

```bash
# Compare metrics between current and a tag
dvc metrics diff v1.0

# Compare parameters
dvc params diff v1.0

# Show metrics for current version
dvc metrics show
```

Output example:

```
Path                    Metric    Old      New      Change
metrics/eval_metrics    mAP       0.42     0.48     0.06
metrics/eval_metrics    mAP50     0.58     0.65     0.07
metrics/eval_metrics    mAP75     0.35     0.41     0.06
```

## Experiment Tracking with DVC

DVC has built-in experiment tracking that does not require additional tools.

### Running Experiments

```bash
# Run an experiment with modified parameters
dvc exp run --set-param train.learning_rate=0.0005

# Run multiple experiments in parallel
dvc exp run --set-param train.learning_rate=0.01 --queue
dvc exp run --set-param train.learning_rate=0.001 --queue
dvc exp run --set-param train.learning_rate=0.0001 --queue
dvc exp run --run-all --parallel 3

# List experiments
dvc exp show

# Compare experiments
dvc exp diff exp-abc123 exp-def456
```

### Promoting an Experiment

```bash
# Apply the best experiment to the workspace
dvc exp apply exp-abc123

# Create a Git branch from an experiment
dvc exp branch exp-abc123 best-lr-experiment
```

## Integration with Git

DVC is designed to work alongside Git. Here is the typical workflow:

```bash
# 1. Make changes to data or code
# 2. Run the pipeline
dvc repro

# 3. Add DVC-tracked changes
dvc push

# 4. Commit metafiles to Git
git add dvc.lock params.yaml metrics/ *.dvc
git commit -m "Update model with new dataset version"
git push

# When a teammate clones or pulls:
git pull
dvc pull  # Downloads the data matching this Git commit
```

### .gitignore Management

DVC automatically manages `.gitignore` entries:

```gitignore
# Auto-added by DVC
/data/coco
/models/best_model.pt
```

### Recommended Git Workflow

```bash
# Always commit .dvc files, dvc.yaml, dvc.lock, params.yaml
# Never commit the actual data files
# Always push DVC data before pushing Git commits

dvc push && git push
```

## Collaboration Workflows

### Onboarding a New Team Member

```bash
# Clone the repository
git clone https://github.com/team/cv-project.git
cd cv-project

# Pull all DVC-tracked data
dvc pull

# Now the workspace has all data, models, and metrics
```

### Working on a Feature Branch

```bash
# Create a feature branch
git checkout -b feature/new-augmentation

# Make changes, run pipeline
dvc repro

# Push data and code
dvc push
git add -A
git commit -m "Add new augmentation strategy"
git push origin feature/new-augmentation
```

## Large Dataset Management

### Handling Datasets That Do Not Fit on Disk

```bash
# Use external storage without copying to workspace
dvc import-url s3://large-dataset/coco-2017.tar.gz data/
dvc import https://github.com/team/data-repo data/annotations

# Fetch only specific files
dvc pull data/train.dvc  # Only download training data
```

### Cache Management

```bash
# Check cache size
du -sh .dvc/cache

# Garbage collect unused cache entries
dvc gc --workspace

# Remove all cached data
dvc gc --all-commits --cloud
```

## Python API

DVC also provides a Python API for programmatic access:

```python
import dvc.api

# Get the URL of a tracked file
url = dvc.api.get_url("data/coco/", repo="https://github.com/team/cv-project")

# Read parameters
params = dvc.api.params_show()
learning_rate = params["train"]["learning_rate"]

# Open a tracked file
with dvc.api.open("data/annotations.json", repo=".") as f:
    import json
    annotations = json.load(f)
```

## Best Practices

1. **Track data early**: Run `dvc add` as soon as data enters the project.
2. **Use pipelines** (`dvc.yaml`) for reproducible training workflows.
3. **Push data before Git**: Always `dvc push` before `git push`.
4. **Use `params.yaml`** for all configurable values in pipelines.
5. **Tag releases** in Git to create named data/model versions.
6. **Use `.dvcignore`** to exclude temporary or intermediate files.
7. **Set up remote storage** before the first team member joins.
8. **Use `dvc gc`** periodically to clean the local cache.
9. **Combine with Git hooks** to automate `dvc push` on commit.
10. **Document the DVC workflow** so all team members follow the same process.

## Summary

DVC brings the discipline of version control to the data and model files that define ML projects. By tracking large files with lightweight metafiles, storing data on any cloud provider, and defining reproducible pipelines, DVC ensures that experiments are reproducible, datasets are versioned, and collaboration is frictionless. When combined with Git, it provides a complete version control solution for the entire ML lifecycle.

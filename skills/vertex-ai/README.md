# Vertex AI Skill

## Purpose

This skill teaches Claude how to run CV/ML training on Google Cloud Vertex AI using the
**custom-container** pattern: package training code into a Docker image, push it to
Artifact Registry, and submit it as a Vertex CustomJob on a GPU machine, reading data
from and writing checkpoints back to GCS. The same container runs locally and in the
cloud — no Vertex-specific training code.

## When to Use

- Submitting a custom-container training job to Vertex AI
- Launching a GPU training run in the cloud
- Staging datasets and artifacts in GCS for a training run
- Hyperparameter tuning on Vertex
- Orchestrating a multi-step Vertex Pipeline (download → train → evaluate → register)

## Key Patterns

- `aiplatform.init` + `CustomContainerTrainingJob` / `CustomJob`
- Reading `AIP_MODEL_DIR` and other `AIP_*` env vars for output paths
- Accelerator/machine sizing (A100 for training, T4 for smoke tests)
- Passing secrets (W&B key) via `environment_variables`
- `HyperparameterTuningJob` + `hypertune` metric reporting
- KFP `@dsl.pipeline` + `PipelineJob` for multi-step flows

Pairs with `gcp` (Artifact Registry, GCS, auth), `docker-cv` (the training image), and
`wandb` (tracking).

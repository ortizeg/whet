# Vertex AI

The vertex-ai skill covers running CV/ML training on Google Cloud Vertex AI using the custom-container pattern — the same Docker image runs locally and in the cloud, with no Vertex-specific training code.

**Skill directory:** `skills/vertex-ai/`

## Purpose

This skill teaches Claude Code to package training code into a Docker image, push it to Artifact Registry, and submit it as a Vertex CustomJob on a GPU machine that reads data from and writes checkpoints back to GCS. It reflects the custom-container training workflow used in production CV pipelines, keeping the local and cloud paths identical.

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

## Anti-Patterns

- Vertex-specific training code that diverges from the local path
- Hardcoding output paths instead of `AIP_MODEL_DIR`
- Baking API keys into the training image
- Reading a large detection dataset file-by-file from `gs://` (copy to SSD first)

# AWS SageMaker Skill

The AWS SageMaker Skill provides expert patterns for building ML training and deployment pipelines on Amazon SageMaker, covering PyTorch estimators, training script entry points, real-time inference endpoints, batch transform jobs, hyperparameter tuning, SageMaker Pipelines for end-to-end orchestration, and S3 data management.

## Purpose

When you need to train models at scale on AWS infrastructure or deploy models behind managed endpoints, SageMaker is the standard managed ML platform. This skill encodes best practices for structuring SageMaker projects: proper estimator configuration, custom inference handlers, pipeline definitions with conditional model registration, and local mode testing to validate scripts before submitting expensive cloud jobs.

## When to Use

- When training models on AWS GPU instances (ml.g5, ml.p4, ml.trn1).
- When deploying models as real-time SageMaker endpoints with auto-scaling.
- When building automated ML pipelines with preprocessing, training, evaluation, and model registration.
- When running hyperparameter tuning jobs to optimize model performance.
- When processing large datasets with SageMaker Processing jobs.

## Key Features

- **PyTorch Estimator patterns** — proper configuration with distributed training support, instance selection, and hyperparameter passing.
- **Training scripts** — SageMaker-compatible entry points that read `SM_CHANNEL_*` and `SM_MODEL_DIR` environment variables.
- **Custom inference handlers** — `model_fn`, `input_fn`, `predict_fn`, `output_fn` for flexible serving.
- **SageMaker Pipelines** — end-to-end workflow orchestration with conditional model registration based on metrics.
- **Hyperparameter tuning** — Bayesian optimization with metric definitions and parameter ranges.
- **Local mode testing** — validate training scripts with `instance_type="local"` before cloud submission.

## Related Skills

- **[PyTorch Lightning](../pytorch-lightning/)** — LightningModule patterns used inside SageMaker training scripts.
- **[Docker CV](../docker-cv/)** — custom training container images when default SageMaker images are insufficient.
- **[W&B](../wandb/)** / **[MLflow](../mlflow/)** — experiment tracking inside SageMaker training containers.
- **[GCP](../gcp/)** — alternative cloud platform patterns for comparison.

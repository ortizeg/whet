# PyTorch Lightning Integration and Autolog

Scope: using `MLFlowLogger` with a Lightning `Trainer`, and enabling framework autologging.

## Integration with PyTorch Lightning

MLflow integrates with PyTorch Lightning through `MLFlowLogger`:

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger

# Create MLflow logger
logger = MLFlowLogger(
    experiment_name="object-detection",
    run_name="lightning-experiment-001",
    tracking_uri="http://mlflow-server:5000",
    log_model=True,
)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="auto",
)

# Train
trainer.fit(model, datamodule=dm)

# Access run info
print(f"Run ID: {logger.run_id}")
print(f"Experiment ID: {logger.experiment_id}")
```

## Autolog

`mlflow.autolog()` (or the framework-specific variant) captures params, metrics, and
the model with no explicit logging calls — the fastest way to instrument an existing
training script.

```python
# Autolog for PyTorch
import mlflow
mlflow.pytorch.autolog()

with mlflow.start_run():
    trainer.fit(model, datamodule=dm)
```

Autolog and explicit logging compose: turn autolog on for the framework-standard
metrics, then add `mlflow.log_metric` calls for anything project-specific.

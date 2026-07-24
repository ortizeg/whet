# Scalars, Grouped Metrics, and HParams

Scope: logging scalar metrics over time, overlaying related metrics on one chart, and comparing runs with the HParams plugin.

## Contents

- [Basic Scalar Logging](#basic-scalar-logging)
- [Logging Multiple Scalars Together](#logging-multiple-scalars-together)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Running Multiple Hyperparameter Experiments](#running-multiple-hyperparameter-experiments)

Scalars are the most common type of data logged to TensorBoard. They track metrics over time.

## Basic Scalar Logging

```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter(log_dir="logs/experiment_001")

# Training loop
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_map = evaluate(model, val_loader)

    # Log scalars
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Metrics/mAP", val_map, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

# Always close the writer
writer.close()
```

## Logging Multiple Scalars Together

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log multiple scalars on the same plot
writer.add_scalars("Loss", {
    "train": train_loss,
    "val": val_loss,
}, epoch)

writer.add_scalars("Per_Class_AP", {
    "car": 0.52,
    "person": 0.48,
    "bike": 0.35,
}, epoch)
```

## Hyperparameter Tuning

TensorBoard's HParams plugin allows comparing runs across hyperparameter configurations.

```python
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

writer = SummaryWriter(log_dir="logs/hparams/run_001")

# Define hyperparameters and metrics
hparam_dict = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "optimizer": "AdamW",
    "dropout": 0.1,
}
metric_dict = {
    "hparam/mAP": best_map,
    "hparam/val_loss": best_val_loss,
}

# Log hyperparameters with their resulting metrics
writer.add_hparams(hparam_dict, metric_dict)
writer.close()
```

## Running Multiple Hyperparameter Experiments

```python
from torch.utils.tensorboard import SummaryWriter

learning_rates = [1e-2, 1e-3, 1e-4]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        run_name = f"lr_{lr}_bs_{bs}"
        writer = SummaryWriter(log_dir=f"logs/hparams/{run_name}")

        # Train with these hyperparameters
        best_map = train_and_evaluate(lr=lr, batch_size=bs)

        writer.add_hparams(
            {"learning_rate": lr, "batch_size": bs},
            {"hparam/mAP": best_map},
        )
        writer.close()
```

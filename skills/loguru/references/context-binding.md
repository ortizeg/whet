# Structured Context Binding for ML

Scope: attaching structured context (epoch, batch, metrics) to log records with
`logger.bind()` so every downstream message carries it automatically.

## Binding context in a training loop

Bind context variables to the logger so every subsequent message includes them
automatically. Bind once per scope rather than repeating the same fields in each call.

```python
from loguru import logger


def train_epoch(epoch: int, dataloader, model, optimizer) -> float:
    """Train one epoch with structured logging."""
    epoch_logger = logger.bind(epoch=epoch)
    epoch_logger.info("Epoch started")

    for batch_idx, batch in enumerate(dataloader):
        loss = train_step(model, optimizer, batch)

        if batch_idx % 100 == 0:
            epoch_logger.bind(batch=batch_idx, loss=f"{loss:.4f}").info(
                "Step {batch} — loss: {loss}",
                batch=batch_idx,
                loss=f"{loss:.4f}",
            )

    avg_loss = compute_average_loss()
    epoch_logger.bind(avg_loss=f"{avg_loss:.4f}").info("Epoch complete")
    return avg_loss
```

## Logging metrics

```python
from loguru import logger


def log_metrics(epoch: int, metrics: dict[str, float]) -> None:
    """Log training metrics with structured context."""
    logger.bind(**{k: f"{v:.4f}" for k, v in metrics.items()}).info(
        "Epoch {epoch} metrics: {metrics}",
        epoch=epoch,
        metrics={k: f"{v:.4f}" for k, v in metrics.items()},
    )


# Usage
log_metrics(epoch=10, metrics={"loss": 0.0234, "accuracy": 0.9512, "lr": 1e-4})
```

Bound fields land in the `extra` dict of the record, so a sink configured with
`serialize=True` emits them as first-class JSON keys — which is what makes bound context
queryable in a log aggregator instead of buried in message text.

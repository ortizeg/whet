# Loguru Sink Configuration

Scope: every sink type loguru supports — stderr, rotating files, JSON-serialized files for
log aggregation, time-based rotation, and custom sink functions — plus per-sink filtering
and custom log levels.

## Contents

- [Stderr (Default)](#stderr-default)
- [Rotating File](#rotating-file)
- [JSON File for Log Aggregation](#json-file-for-log-aggregation)
- [Time-Based Rotation](#time-based-rotation)
- [Custom Sink Function](#custom-sink-function)
- [Per-Module Filtering](#per-module-filtering)
- [Custom Filter Function](#custom-filter-function)
- [Custom Levels for ML](#custom-levels-for-ml)

## Stderr (Default)

```python
logger.add(sys.stderr, level="INFO", colorize=True)
```

## Rotating File

```python
# Rotate when file exceeds 100 MB, keep 7 days, compress old logs
logger.add(
    "logs/training.log",
    rotation="100 MB",
    retention="7 days",
    compression="gz",
)
```

## JSON File for Log Aggregation

`serialize=True` emits one JSON object per line, which is what CloudWatch, Elasticsearch,
and GCP Cloud Logging expect. Use it for anything running in production.

```python
# JSON-serialized logs for Elasticsearch, CloudWatch, or GCP Cloud Logging
logger.add(
    "logs/training.json",
    serialize=True,
    rotation="500 MB",
    retention="30 days",
)
```

## Time-Based Rotation

```python
# New log file every day at midnight
logger.add("logs/training_{time}.log", rotation="00:00", retention="30 days")
```

## Custom Sink Function

A sink can be any callable — use this to forward log records into an experiment tracker or
a metrics backend.

```python
def wandb_sink(message: str) -> None:
    """Forward log messages to Weights & Biases."""
    import wandb

    record = message.record
    if record["level"].name == "INFO":
        wandb.log({"log": record["message"]})


logger.add(wandb_sink, level="INFO")
```

## Per-Module Filtering

```python
# Suppress noisy libraries
logger.add(sys.stderr, level="WARNING", filter="PIL")
logger.add(sys.stderr, level="WARNING", filter="matplotlib")
logger.add(sys.stderr, level="INFO", filter="my_project")
```

## Custom Filter Function

```python
def no_health_checks(record):
    """Filter out health check log spam."""
    return "/health" not in record["message"]


logger.add(sys.stderr, filter=no_health_checks)
```

## Custom Levels for ML

Custom levels make metric and checkpoint events greppable and independently filterable.

```python
from loguru import logger

# Add custom levels for ML-specific events
logger.level("METRIC", no=25, color="<yellow>", icon="@")
logger.level("CHECKPOINT", no=25, color="<magenta>", icon="*")


def log_metric(name: str, value: float, step: int) -> None:
    logger.log("METRIC", "{name}={value:.4f} step={step}", name=name, value=value, step=step)


def log_checkpoint(path: str, epoch: int) -> None:
    logger.log("CHECKPOINT", "Saved {path} at epoch {epoch}", path=path, epoch=epoch)
```

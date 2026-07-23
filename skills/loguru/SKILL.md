---
name: loguru
description: >
  Use this skill when adding or standardizing logging in a CV/ML project — configuring
  Loguru sinks, structured JSON logs, log levels, context binding, file rotation, and
  intercepting stdlib logging. Reach for it any time code has print() statements or
  stdlib logging calls that should become structured logs, or any time a new module needs
  a logger, even if the user doesn't say "Loguru" — it is the mandatory logging
  convention in this project.
---

# Loguru Skill

Structured logging for CV/ML projects using loguru. This is a mandatory project convention — all repositories use loguru instead of stdlib `logging` or `print()`.

## Why Loguru over stdlib logging

stdlib `logging` needs per-module boilerplate (create logger, configure handlers, formatters, propagation). Loguru replaces it with one import and sensible defaults:

```python
from loguru import logger

logger.info("Training started")  # zero setup
```

Advantages that matter for ML: structured context binding (attach epoch/loss to all messages), tracebacks with full variable values, built-in rotation/retention, JSON serialization for aggregation, thread/process safety with DataLoader workers, and lazy formatting (`logger.info("Loss: {}", loss)` only formats when the level is active).

## Standard Project Setup

Every project includes a `setup_logging()` function called once at entry point.

```python
# src/{{package_name}}/log.py
"""Logging configuration."""

from __future__ import annotations

import sys

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    serialize: bool = False,
) -> None:
    """Configure loguru for the project.

    Call once at application entry point (train.py, serve.py, cli.py).
    """
    # Remove the default stderr handler
    logger.remove()

    # Stderr handler with color and concise format
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Optional file handler with rotation
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            serialize=serialize,
        )
```

### Entry Point Usage

```python
# src/{{package_name}}/train.py
"""Training entry point."""

from loguru import logger

from my_project.log import setup_logging


def main() -> None:
    setup_logging(level="DEBUG", log_file="logs/training.log")
    logger.info("Starting training")
    # ...


if __name__ == "__main__":
    main()
```

## Structured Logging for ML

Bind context variables to the logger so every subsequent message includes them automatically.

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

### Logging Metrics

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

## Log Sinks

### Stderr (Default)

```python
logger.add(sys.stderr, level="INFO", colorize=True)
```

### Rotating File

```python
# Rotate when file exceeds 100 MB, keep 7 days, compress old logs
logger.add(
    "logs/training.log",
    rotation="100 MB",
    retention="7 days",
    compression="gz",
)
```

### JSON File for Log Aggregation

```python
# JSON-serialized logs for Elasticsearch, CloudWatch, or GCP Cloud Logging
logger.add(
    "logs/training.json",
    serialize=True,
    rotation="500 MB",
    retention="30 days",
)
```

### Time-Based Rotation

```python
# New log file every day at midnight
logger.add("logs/training_{time}.log", rotation="00:00", retention="30 days")
```

### Custom Sink Function

```python
def wandb_sink(message: str) -> None:
    """Forward log messages to Weights & Biases."""
    import wandb

    record = message.record
    if record["level"].name == "INFO":
        wandb.log({"log": record["message"]})


logger.add(wandb_sink, level="INFO")
```

## Integration with PyTorch Lightning

Route all Lightning logs through loguru using an intercept handler.

```python
# src/{{package_name}}/log.py
import logging

from loguru import logger


class InterceptHandler(logging.Handler):
    """Route stdlib logging through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# In setup_logging(), after configuring your loguru sinks (see Standard Project
# Setup above), route all stdlib logging through loguru with one line:
#   logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

### Lightning Trainer

```python
import lightning as L
from loguru import logger

from my_project.log import setup_logging


def main() -> None:
    setup_logging(level="INFO", log_file="logs/training.log")

    trainer = L.Trainer(
        max_epochs=50,
        enable_progress_bar=True,
    )

    logger.info("Trainer configured, starting fit")
    trainer.fit(model, datamodule)
```

## Integration with Third-Party Libraries

Intercept stdlib logging from any library (uvicorn, torch, transformers, etc.).

```python
import logging

from my_project.log import InterceptHandler

# Route specific libraries through loguru
for lib in ["uvicorn", "uvicorn.access", "torch", "transformers"]:
    logging.getLogger(lib).handlers = [InterceptHandler()]
```

### FastAPI / Uvicorn

```python
import uvicorn

from my_project.log import setup_logging

setup_logging(level="INFO")

uvicorn.run(
    "my_project.api:app",
    host="0.0.0.0",
    port=8000,
    log_config=None,  # Disable uvicorn's default logging — loguru handles it
)
```

## Exception Handling

### Decorator

```python
from loguru import logger


@logger.catch(reraise=True)
def train_step(model, optimizer, batch):
    """Train step with automatic exception logging."""
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()
    return loss.item()
```

### Context Manager

```python
from loguru import logger


def process_batch(batch):
    with logger.catch(message="Failed to process batch"):
        result = model.predict(batch)
        return result
```

### Exception with Full Context

```python
from loguru import logger


def load_checkpoint(path: str):
    try:
        checkpoint = torch.load(path)
    except Exception:
        logger.exception("Failed to load checkpoint from {}", path)
        raise
```

Loguru's `logger.exception()` includes the full traceback with variable values at each frame — far more useful for debugging than stdlib's traceback.

## Filtering and Levels

### Per-Module Filtering

```python
# Suppress noisy libraries
logger.add(sys.stderr, level="WARNING", filter="PIL")
logger.add(sys.stderr, level="WARNING", filter="matplotlib")
logger.add(sys.stderr, level="INFO", filter="my_project")
```

### Custom Filter Function

```python
def no_health_checks(record):
    """Filter out health check log spam."""
    return "/health" not in record["message"]


logger.add(sys.stderr, filter=no_health_checks)
```

### Custom Levels for ML

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

## Pydantic Configuration

```python
from pydantic import BaseModel, Field


class LogConfig(BaseModel, frozen=True):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level (DEBUG, INFO, WARNING, ERROR)")
    format: str = Field(
        default=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        description="Log format string",
    )
    log_file: str | None = Field(default=None, description="Path to log file")
    rotation: str = Field(default="100 MB", description="Log file rotation size")
    retention: str = Field(default="7 days", description="Log file retention period")
    serialize: bool = Field(default=False, description="Serialize logs as JSON")
    colorize: bool = Field(default=True, description="Colorize stderr output")
```

```yaml
# configs/logging.yaml — Hydra-compatible
log:
  level: INFO
  log_file: logs/training.log
  rotation: 100 MB
  retention: 7 days
  serialize: false
  colorize: true
```

Then `setup_logging(config: LogConfig)` calls `logger.remove()` and adds the stderr
(and optional file) sinks using `config.level`, `config.format`, `config.rotation`, etc.

## Dependency

loguru is a standard project dependency: `pixi add "loguru>=0.7"`.

## Best Practices

1. **Call `setup_logging()` once at entry point** — never in library modules, only in `main()`, `train.py`, `serve.py`, or CLI commands.
2. **Use `from loguru import logger`** — never `logging.getLogger()` or `print()` in project code.
3. **Use lazy formatting** — `logger.info("Loss: {}", loss)` not `logger.info(f"Loss: {loss}")` to avoid formatting overhead when the level is filtered.
4. **Bind context, don't repeat it** — use `logger.bind(epoch=epoch)` to attach context to all messages within a scope.
5. **Intercept stdlib logging** — use `InterceptHandler` so third-party libraries (Lightning, uvicorn, torch) route through loguru.
6. **Rotate and retain log files** — always set `rotation` and `retention` to prevent unbounded disk usage.
7. **Use `serialize=True` for production** — JSON logs are required for log aggregation (CloudWatch, Elasticsearch, GCP Cloud Logging).
8. **Use `logger.catch` for fault tolerance** — decorate functions that must not crash silently.
9. **Filter noisy libraries** — set `level="WARNING"` for PIL, matplotlib, and other verbose libraries.
10. **Never log secrets** — do not log API keys, tokens, or credentials; redact sensitive fields.

## Anti-Patterns to Avoid

- ❌ Using `print()` for debugging or status messages — use `logger.debug()` or `logger.info()` instead.
- ❌ Using `logging.getLogger(__name__)` — use `from loguru import logger` everywhere.
- ❌ Calling `setup_logging()` in library modules — only call it in entry points; library code just imports `logger`.
- ❌ Using f-strings in log calls — `logger.info(f"Loss: {loss}")` formats even when the level is filtered; use `logger.info("Loss: {}", loss)`.
- ❌ Adding handlers in multiple places — remove the default handler once with `logger.remove()`, then add your sinks in one place.
- ❌ Logging large tensors or arrays — log shapes and summary statistics, not the full data: `logger.debug("Batch shape: {}", batch.shape)`.
- ❌ Ignoring log file rotation — unrotated logs on training nodes fill disks and crash jobs.
- ❌ Logging at DEBUG level in production — use INFO or WARNING; DEBUG is for development only.

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

Structured logging for CV/ML projects using loguru. **This is a mandatory project
convention**: all repositories use loguru instead of stdlib `logging` or `print()`.
This page holds the rule and the setup you need every time; the deep dives below cover
sinks, context binding, stdlib interception, and config validation.

## The rule

**Never use `print()`. Never use `logging.getLogger(__name__)`.** In every module —
library code included — the only logging import is:

```python
from loguru import logger

logger.info("Training started")  # zero setup, works everywhere
```

stdlib `logging` needs per-module boilerplate (create logger, configure handlers,
formatters, propagation). Loguru replaces it with one import and sensible defaults, plus
the things that matter for ML: structured context binding (attach epoch/loss to all
messages), tracebacks with full variable values, built-in rotation/retention, JSON
serialization for aggregation, thread/process safety with DataLoader workers, and lazy
formatting (`logger.info("Loss: {}", loss)` only formats when the level is active).

## Standard project setup

Every project includes a `setup_logging()` function called **once** at the entry point —
never in library modules.

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

### Entry point usage

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

### Binding context

Attach fields once and every message in scope carries them:

```python
epoch_logger = logger.bind(epoch=epoch)
epoch_logger.info("Epoch started")
epoch_logger.bind(loss=f"{loss:.4f}").info("Step done — loss: {loss}", loss=f"{loss:.4f}")
```

## Conventions

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

## Anti-patterns

- ❌ Using `print()` for debugging or status messages — use `logger.debug()` or `logger.info()` instead.
- ❌ Using `logging.getLogger(__name__)` — use `from loguru import logger` everywhere.
- ❌ Calling `setup_logging()` in library modules — only call it in entry points; library code just imports `logger`.
- ❌ Using f-strings in log calls — `logger.info(f"Loss: {loss}")` formats even when the level is filtered; use `logger.info("Loss: {}", loss)`.
- ❌ Adding handlers in multiple places — remove the default handler once with `logger.remove()`, then add your sinks in one place.
- ❌ Logging large tensors or arrays — log shapes and summary statistics, not the full data: `logger.debug("Batch shape: {}", batch.shape)`.
- ❌ Ignoring log file rotation — unrotated logs on training nodes fill disks and crash jobs.
- ❌ Logging at DEBUG level in production — use INFO or WARNING; DEBUG is for development only.

## Deep dives

- `references/sink-configuration.md` — read when adding a sink beyond the defaults: JSON/`serialize=True` for log aggregation, time-based rotation, custom sink functions, per-module filters, or custom ML log levels.
- `references/context-binding.md` — read when adding structured context to a training loop or emitting metrics as queryable fields rather than message text.
- `references/intercepting-stdlib.md` — read when a third-party library (Lightning, uvicorn, torch, transformers) is still logging through stdlib and needs to route into loguru.
- `references/exception-handling.md` — read when wiring `@logger.catch`, the catch context manager, or `logger.exception()` for full-variable tracebacks.
- `references/pydantic-log-config.md` — read when logging settings must be validated as a Pydantic model or composed from a Hydra/YAML config.

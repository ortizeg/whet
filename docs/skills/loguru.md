# Loguru

The Loguru skill establishes loguru as the standard logging library for all CV/ML projects, replacing stdlib `logging` and `print()`.

**Skill directory:** `skills/loguru/`

## Purpose

ML projects produce dense, structured output: per-epoch metrics, batch-level loss, checkpoint events, evaluation results. stdlib `logging` requires boilerplate in every module (getLogger, handlers, formatters). Loguru replaces all of that with a single import and pre-configured defaults, plus structured context binding, lazy formatting, and built-in file rotation.

## When to Use

- Every project -- loguru is a mandatory convention, not optional
- Any module that needs logging -- `from loguru import logger`
- Entry points (train.py, serve.py) -- call `setup_logging()` once
- Routing third-party library output through a unified logger

## Key Patterns

### Standard Setup

```python
# src/my_project/log.py
import sys
from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    if log_file:
        logger.add(log_file, rotation="100 MB", retention="7 days", compression="gz")
```

### Structured Context for Training

```python
from loguru import logger

epoch_logger = logger.bind(epoch=5)
epoch_logger.info("Training epoch started")
epoch_logger.bind(loss="0.0234").info("Step complete")
```

### Intercept stdlib Logging

```python
import logging
from loguru import logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        level = logger.level(record.levelname).name
        logger.opt(depth=2, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

## Anti-Patterns to Avoid

- Do not use `print()` for status messages -- use `logger.info()`
- Do not use `logging.getLogger(__name__)` -- use `from loguru import logger`
- Do not call `setup_logging()` in library modules -- only in entry points
- Do not use f-strings in log calls -- use `logger.info("Loss: {}", loss)` for lazy formatting
- Do not skip file rotation -- unrotated logs fill disks and crash training jobs

## Combines Well With

- **Master Skill** -- loguru is a standard dependency in all generated projects
- **PyTorch Lightning** -- InterceptHandler routes Lightning logs through loguru
- **Pydantic** -- LogConfig model for type-safe logging configuration
- **Hydra Config** -- logging settings in `configs/logging.yaml`

## Full Reference

See [`skills/loguru/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/loguru/SKILL.md) for complete patterns including custom ML log levels, JSON serialization, Pydantic configuration, and exception handling decorators.

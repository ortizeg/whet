# Logging Convention

Scope: the mandatory loguru setup every generated project ships with, and where
`setup_logging()` may and may not be called.

## Loguru Everywhere

All projects use **loguru** for logging. Never use `print()` or `logging.getLogger()`.

```python
# ✅ Correct — every module
from loguru import logger

logger.info("Training started")
logger.debug("Batch shape: {}", batch.shape)

# ❌ Wrong — never use these
print("Training started")            # No print
import logging                        # No stdlib logging
log = logging.getLogger(__name__)     # No getLogger
```

The Ruff `T20` rule set (flake8-print) is enabled in the generated `pyproject.toml`
specifically to make `print()` a lint failure rather than a convention people forget.

## setup_logging()

Every project includes a `setup_logging()` function. Call it **once** at the entry point,
never in library modules.

```python
# src/{{package_name}}/log.py
import sys
from loguru import logger


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru. Call once in main(), train.py, or serve.py."""
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    if log_file:
        logger.add(log_file, rotation="100 MB", retention="7 days", compression="gz")
```

Entry points that call it: `train.py`, `serve.py`, CLI `main()`. Library modules only
`from loguru import logger` and log — configuring sinks inside a library duplicates handlers
for anyone importing it.

See the **Loguru** skill for structured context binding, InterceptHandler for third-party
libraries, JSON serialization, and Pydantic configuration.

# Intercepting Stdlib Logging

Scope: routing stdlib `logging` output from third-party libraries (PyTorch Lightning,
uvicorn, torch, transformers) through loguru with an `InterceptHandler`.

## Contents

- [The InterceptHandler](#the-intercepthandler)
- [PyTorch Lightning](#pytorch-lightning)
- [Third-Party Libraries](#third-party-libraries)
- [FastAPI / Uvicorn](#fastapi--uvicorn)

## The InterceptHandler

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


# In setup_logging(), after configuring your loguru sinks, route all stdlib logging
# through loguru with one line:
#   logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
```

## PyTorch Lightning

With the intercept handler installed in `setup_logging()`, Lightning's own logging flows
into your loguru sinks with no further configuration.

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

## Third-Party Libraries

To target specific libraries rather than the root logger, replace their handlers directly.

```python
import logging

from my_project.log import InterceptHandler

# Route specific libraries through loguru
for lib in ["uvicorn", "uvicorn.access", "torch", "transformers"]:
    logging.getLogger(lib).handlers = [InterceptHandler()]
```

## FastAPI / Uvicorn

Uvicorn installs its own logging config on startup; pass `log_config=None` so loguru stays
in charge.

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

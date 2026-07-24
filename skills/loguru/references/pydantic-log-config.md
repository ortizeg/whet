# Pydantic-Validated Log Configuration

Scope: expressing logging settings as a frozen Pydantic model so they can be validated at
load time and composed from a Hydra/YAML config.

## LogConfig model

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

## Hydra-compatible YAML

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

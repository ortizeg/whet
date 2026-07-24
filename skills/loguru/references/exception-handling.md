# Exception Handling with Loguru

Scope: capturing exceptions with `@logger.catch`, the `logger.catch` context manager, and
`logger.exception()` for full-variable tracebacks.

## Decorator

Decorate functions that must not fail silently. `reraise=True` logs the traceback and then
lets the exception propagate.

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

## Context Manager

```python
from loguru import logger


def process_batch(batch):
    with logger.catch(message="Failed to process batch"):
        result = model.predict(batch)
        return result
```

## Exception with Full Context

```python
from loguru import logger


def load_checkpoint(path: str):
    try:
        checkpoint = torch.load(path)
    except Exception:
        logger.exception("Failed to load checkpoint from {}", path)
        raise
```

Loguru's `logger.exception()` includes the full traceback with variable values at each
frame — far more useful for debugging than stdlib's traceback.

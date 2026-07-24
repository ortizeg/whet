# Fixing Common Violations

Scope: the specific code rewrites that clear the lint rules most often hit in CV/ML code —
`T20` (print), `PTH` (os.path), and `EM`/`TRY`-style inline raise messages.

## Contents

- [Logging instead of print (T20)](#logging-instead-of-print-t20)
- [Path handling with pathlib (PTH)](#path-handling-with-pathlib-pth)
- [Error messages](#error-messages)

## Logging instead of print (T20)

The `T20` rule bans `print()` in source. Use a logger with lazy `%`-style args
(the project convention is loguru — see the loguru skill):

```python
from loguru import logger

logger.info("Starting epoch {} with {} batches", epoch, num_batches)
logger.debug("Batch {}/{}, loss={:.4f}", batch_idx, num_batches, loss)
```

## Path handling with pathlib (PTH)

```python
from __future__ import annotations

from pathlib import Path


def find_images(data_dir: Path, extensions: tuple[str, ...] = (".jpg", ".png")) -> list[Path]:
    """Find all image files in a directory recursively."""
    images: list[Path] = []
    for ext in extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
    return sorted(images)


# CORRECT: pathlib
output_dir = Path("results") / "experiment_1"
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / "model.pt"

# WRONG: os.path
import os
output_dir = os.path.join("results", "experiment_1")
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "model.pt")
```

## Error messages

```python
# CORRECT: Assign message to variable, raise with variable
msg = f"Expected 3-channel image, got {image.shape[-1]} channels"
raise ValueError(msg)

# WRONG: Inline string in raise
raise ValueError(f"Expected 3-channel image, got {image.shape[-1]} channels")

# WRONG: No f-string when dynamic content is needed
raise ValueError("Invalid image shape")
```

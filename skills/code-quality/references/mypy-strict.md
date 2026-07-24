# MyPy Strict Mode and Type Hint Rules

Scope: what strict mode buys you, when the untyped-library override is justified, and the
annotation patterns every file is expected to follow.

## Contents

- [Why Strict Mode?](#why-strict-mode)
- [Always annotate function signatures](#always-annotate-function-signatures)
- [Use modern type syntax (Python 3.11+)](#use-modern-type-syntax-python-311)
- [Use `from __future__ import annotations` in every file](#use-from-__future__-import-annotations-in-every-file)
- [Type hint class attributes](#type-hint-class-attributes)
- [Use Protocol for structural typing](#use-protocol-for-structural-typing)
- [Use TypeAlias for complex types](#use-typealias-for-complex-types)
- [Running MyPy](#running-mypy)

## Why Strict Mode?

Strict mode enables all of mypy's strictness flags at once. This catches:

- Functions without type annotations
- Variables with implicit `Any` types
- Missing return type annotations
- Untyped decorators
- Implicit `Optional` types

The override section in the standard config exempts third-party libraries that do not ship
type stubs. This list should be kept as small as possible -- if a library provides stubs,
remove it from the override.

## Always annotate function signatures

```python
# CORRECT: Fully annotated
def process_image(
    image: np.ndarray,
    target_size: tuple[int, int],
    normalize: bool = True,
) -> np.ndarray:
    """Process a single image for model input."""
    ...

# WRONG: Missing annotations
def process_image(image, target_size, normalize=True):
    ...
```

## Use modern type syntax (Python 3.11+)

```python
# CORRECT: built-in generics and PEP 604 unions
def get_labels() -> list[str]: ...
def get_config() -> dict[str, int]: ...
def maybe_transform(image: np.ndarray) -> np.ndarray | None: ...

# WRONG: legacy typing module — List[str], Dict[str, int], Optional[np.ndarray]
```

## Use `from __future__ import annotations` in every file

```python
# ALWAYS include this as the first import
from __future__ import annotations

# This enables PEP 604 union syntax and deferred evaluation
# in all Python 3.11+ files
```

## Type hint class attributes

```python
from __future__ import annotations

import torch.nn as nn


class Detector(nn.Module):
    """Object detection model."""

    backbone: nn.Module
    head: nn.Module
    num_classes: int

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self._build_backbone()
        self.head = self._build_head()
```

## Use Protocol for structural typing

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ImageTransform(Protocol):
    """Protocol for image transformation functions."""

    def __call__(self, image: np.ndarray) -> np.ndarray: ...


def apply_transforms(
    image: np.ndarray,
    transforms: list[ImageTransform],
) -> np.ndarray:
    """Apply a sequence of transforms to an image."""
    for transform in transforms:
        image = transform(image)
    return image
```

## Use TypeAlias for complex types

```python
from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

# Define aliases for frequently used complex types
ImageArray: TypeAlias = np.ndarray
BoundingBox: TypeAlias = tuple[float, float, float, float]
BatchTensor: TypeAlias = torch.Tensor
DetectionList: TypeAlias = list[tuple[BoundingBox, float, int]]
```

## Running MyPy

```bash
mypy src/ --strict                          # type check
mypy src/ --strict --html-report report/    # optional HTML report
```

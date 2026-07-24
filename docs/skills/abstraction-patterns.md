# Abstraction Patterns

The Abstraction Patterns skill defines design patterns and architectural abstractions for scalable ML/CV codebases, using Python's `ABC`, `Protocol`, and composition patterns.

**Skill directory:** `skills/abstraction-patterns/`

## Purpose

ML code often starts as scripts and grows into unmaintainable monoliths. This skill teaches Claude Code to apply software engineering design patterns to ML codebases: abstract base classes for model families, Protocol types for duck-typed interfaces, strategy patterns for interchangeable components (backbones, losses, augmentations), and registry patterns for dynamic component selection.

## When to Use

- Designing model architectures with interchangeable backbones
- Building data pipelines with pluggable transforms
- Creating inference services that support multiple model types
- Any codebase with more than 3 model variants or dataset types

## Key Patterns

### Protocol-Based Interface

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from torch import Tensor

@runtime_checkable
class Backbone(Protocol):
    """Protocol for model backbones."""

    @property
    def output_dim(self) -> int: ...

    def forward(self, x: Tensor) -> Tensor: ...


@runtime_checkable
class ImageTransform(Protocol):
    """Protocol for image augmentation transforms."""

    def __call__(self, image: np.ndarray) -> np.ndarray: ...
```

### Registry Pattern

```python
from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

class Registry(dict[str, Callable[..., T]]):
    """Registry for dynamic component creation."""

    def register(self, name: str) -> Callable:
        def decorator(cls: Callable[..., T]) -> Callable[..., T]:
            self[name] = cls
            return cls
        return decorator

    def build(self, name: str, **kwargs) -> T:
        if name not in self:
            raise KeyError(f"Unknown component: {name}. Available: {list(self.keys())}")
        return self[name](**kwargs)


BACKBONES = Registry()

@BACKBONES.register("resnet50")
class ResNet50Backbone:
    output_dim: int = 2048
    ...

@BACKBONES.register("efficientnet_b0")
class EfficientNetB0Backbone:
    output_dim: int = 1280
    ...

# Usage: backbone = BACKBONES.build("resnet50", pretrained=True)
```

### Strategy Pattern for Losses

```python
from abc import ABC, abstractmethod

class DetectionLoss(ABC):
    """Abstract base class for detection losses."""

    @abstractmethod
    def forward(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute loss components. Returns dict of named losses."""
        ...

class FocalLoss(DetectionLoss):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets) -> dict[str, Tensor]:
        ...
```

## Anti-Patterns to Avoid

- Do not create abstractions prematurely -- wait until you have 2+ concrete implementations
- Avoid deep inheritance hierarchies -- prefer composition and protocols
- Do not use ABCs where Protocols would suffice -- Protocols enable duck typing without coupling
- Avoid god registries that mix unrelated components

## Combines Well With

- **PyTorch Lightning** -- Abstract model interfaces for interchangeable LightningModules
- **Hydra Config** -- Registry-based component instantiation from config
- **Pydantic** -- Validated configuration for registered components
- **Testing** -- Test abstract interfaces with concrete test doubles

## Full Reference

See [`skills/abstraction-patterns/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/abstraction-patterns/SKILL.md) for patterns including factory methods, builder patterns for complex pipelines, and mixin classes for common ML functionality.

# Interface Design with Abstract Base Classes

Scope: designing a small, consistent interface (`update`/`compute`/`reset`) that many implementations share, using metric computation as the worked example.

## Contents

- [When an ABC earns its place](#when-an-abc-earns-its-place)
- [The Metric interface](#the-metric-interface)
- [Composing implementations](#composing-implementations)

## When an ABC earns its place

An abstract base class is justified when several concrete implementations must be
interchangeable at the call site — the training loop should not care whether it is
accumulating accuracy or IoU. If there is only one implementation, skip the ABC and
write the class directly.

Metrics in CV projects require accumulation over batches and reset semantics. Abstract this into a consistent `update`/`compute`/`reset` interface.

## The Metric interface

```python
"""Metric computation with accumulation and reset."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Base class for metrics accumulated over batches; reset() between epochs."""

    @abstractmethod
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        ...

    @abstractmethod
    def compute(self) -> dict[str, float]:
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset accumulated state for a new epoch."""
        ...


class AccuracyMetric(Metric):
    """Top-1 accuracy with accumulation."""

    def __init__(self) -> None:
        self._correct: int = 0
        self._total: int = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        pred_classes = np.argmax(predictions, axis=1)
        self._correct += int(np.sum(pred_classes == targets))
        self._total += len(targets)

    def compute(self) -> dict[str, float]:
        if self._total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": self._correct / self._total}

    def reset(self) -> None:
        self._correct = 0
        self._total = 0


class IoUMetric(Metric):
    """Per-class + mean IoU for segmentation; predictions/targets are (N, H, W) class indices."""

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self._num_classes = num_classes
        self._ignore_index = ignore_index
        self._intersection = np.zeros(num_classes, dtype=np.int64)
        self._union = np.zeros(num_classes, dtype=np.int64)

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        mask = targets != self._ignore_index
        pred_masked, target_masked = predictions[mask], targets[mask]
        for cls in range(self._num_classes):
            pred_cls = pred_masked == cls
            target_cls = target_masked == cls
            self._intersection[cls] += int(np.sum(pred_cls & target_cls))
            self._union[cls] += int(np.sum(pred_cls | target_cls))

    def compute(self) -> dict[str, float]:
        iou_per_class = np.zeros(self._num_classes)
        for cls in range(self._num_classes):
            if self._union[cls] > 0:
                iou_per_class[cls] = self._intersection[cls] / self._union[cls]
        result: dict[str, float] = {"mean_iou": float(np.mean(iou_per_class))}
        for cls in range(self._num_classes):
            result[f"iou_class_{cls}"] = float(iou_per_class[cls])
        return result

    def reset(self) -> None:
        self._intersection[:] = 0
        self._union[:] = 0
```

## Composing implementations

Because every implementation shares one interface, a collection can drive them all
without knowing what they are:

```python
class MetricCollection:
    """Runs several metrics together, prefixing each metric's keys with its name."""

    def __init__(self, metrics: dict[str, Metric]) -> None:
        self._metrics = metrics

    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        for metric in self._metrics.values():
            metric.update(predictions, targets)

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for name, metric in self._metrics.items():
            for key, value in metric.compute().items():
                results[f"{name}/{key}"] = value
        return results

    def reset(self) -> None:
        for metric in self._metrics.values():
            metric.reset()
```

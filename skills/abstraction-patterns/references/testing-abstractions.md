# Testing Abstractions

Scope: pytest examples that verify an abstraction's contract — accumulation, reset semantics, and error paths.

A good abstraction is easy to test in isolation: state accumulates as documented,
`reset()` really clears it, and invalid input raises the documented exception. If a
wrapper is hard to test without real hardware or files, the abstraction boundary is
in the wrong place.

```python
"""Tests for abstraction patterns."""

from __future__ import annotations

import numpy as np
import pytest

from my_project.io import VideoReader, load_image
from my_project.metrics import AccuracyMetric, IoUMetric, MetricCollection


def test_accuracy_metric_accumulates_across_batches() -> None:
    metric = AccuracyMetric()
    metric.update(np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]), np.array([0, 1, 0, 0]))  # 3/4
    metric.update(np.array([[0.8, 0.2], [0.3, 0.7]]), np.array([0, 1]))  # 2/2
    assert metric.compute()["accuracy"] == pytest.approx(5 / 6)


def test_accuracy_metric_reset() -> None:
    metric = AccuracyMetric()
    metric.update(np.array([[0.9, 0.1]]), np.array([0]))
    metric.reset()
    assert metric.compute()["accuracy"] == 0.0


def test_load_image_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image("/nonexistent/image.jpg")
```

---
name: abstraction-patterns
description: >
  Use this skill when deciding whether and how to introduce an abstraction, wrapper,
  base class, or interface in an AI/CV Python codebase — refactoring repeated logic,
  wrapping a third-party API, or judging whether code is over-engineered. Applies the
  Rule of Three and interface-design principles to cut cognitive load without premature
  generalization. Reach for it any time you'd otherwise copy-paste a third variation of
  the same logic or spin up a class for a single function, even if the user doesn't say
  "abstraction". Not for choosing which library to depend on (see library-review) or for
  type/data validation (see pydantic).
---

# Abstraction Patterns Skill

Well-abstracted Python for AI/CV projects. Abstraction exists to reduce cognitive load, not to add layers — every abstraction must justify itself by making at least three call sites simpler.

## The Rule: When to Abstract

Abstract when:
- **Three or more call sites** share the same logic
- **Resource management** requires setup/teardown (video readers, model sessions, database connections)
- **Complex validation** needs to happen consistently (image format checking, bounding box validation)
- **External dependencies** need to be isolated for testing (file I/O, API calls, hardware access)

Do NOT abstract when:
- There is only one call site (inline it)
- The abstraction hides important details (GPU memory management, batch dimension handling)
- A simple function would suffice (do not create a class for a single method)

## Pattern 1: VideoReader Abstraction

Video reading involves resource management (opening/closing file handles), frame iteration, and metadata access — a perfect candidate for abstraction. Raw `cv2.VideoCapture` usage scattered across a codebase is easy to get wrong (forgetting `cap.release()`, missing `isOpened()` checks). Wrap it in a context manager:

```python
"""Video reader with proper resource management."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable metadata for a video file."""

    path: Path
    fps: float
    total_frames: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.total_frames / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Context-managed OpenCV video reader.

    Always use as a context manager so resources are released:
        with VideoReader("input.mp4") as reader:
            for frame in reader:
                process(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        return self._metadata

    def __enter__(self) -> VideoReader:
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")
        self._metadata = VideoMetadata(
            path=self._path,
            fps=self._cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return self

    def __exit__(self, *args: object) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield frame

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Read a specific frame by zero-based index (BGR)."""
        if self._cap is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        if frame_idx < 0 or frame_idx >= self.metadata.total_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.metadata.total_frames})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")
        return frame
```

Usage is now clean and leak-free:

```python
with VideoReader("input.mp4") as reader:
    print(f"{reader.metadata.fps} FPS, {reader.metadata.total_frames} frames")
    for frame in reader:
        visualize(frame, model.predict(frame))
    middle = reader.read_frame(reader.metadata.total_frames // 2)
```

## Pattern 2: Image Loading Abstraction

Image loading seems simple but involves format detection, color space conversion, validation, and error handling. Abstracting it prevents inconsistencies.

```python
"""Robust image loading with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np

ColorSpace = Literal["rgb", "bgr", "gray"]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
})


def load_image(
    path: str | Path,
    color_space: ColorSpace = "rgb",
    max_size: int | None = None,
) -> np.ndarray:
    """Load an image with validation and optional longest-edge resize.

    Returns (H, W, 3) for color or (H, W) for grayscale. Raises
    FileNotFoundError, ValueError (unsupported extension), or RuntimeError
    (decode failure).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)  # OpenCV loads BGR
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")

    if color_space == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_space == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # "bgr" needs no conversion

    if max_size is not None:
        h, w = image.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return image


def save_image(
    image: np.ndarray,
    path: str | Path,
    color_space: ColorSpace = "rgb",
    quality: int = 95,
) -> None:
    """Save an image, creating parent dirs and picking encode params by suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")

    if color_space == "rgb" and image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    params: list[int] = []
    if path.suffix.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    cv2.imwrite(str(path), image, params)
```

## Pattern 3: Metric Computation Abstraction

Metrics in CV projects require accumulation over batches and reset semantics. Abstract this into a consistent `update`/`compute`/`reset` interface.

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

## Pattern 4: Model Inference Wrapper

Wrap model inference to handle preprocessing, batching, postprocessing, and device management in one place.

```python
"""Model inference wrapper with preprocessing and postprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class InferenceWrapper:
    """Runs inference with device management, preprocessing, batching, and postprocessing in one interface."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        input_size: tuple[int, int] = (224, 224),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        self._model = model.to(device).eval()
        self._device = torch.device(device)
        self._input_size = input_size
        self._mean = np.array(mean, dtype=np.float32)
        self._std = np.array(std, dtype=np.float32)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Resize, normalize, CHW, add batch dim, move to device."""
        import cv2

        resized = cv2.resize(image, (self._input_size[1], self._input_size[0]))
        normalized = (resized.astype(np.float32) / 255.0 - self._mean) / self._std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self._device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        return self._model(self.preprocess(image)).cpu().numpy()

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray]) -> np.ndarray:
        batch = torch.cat([self.preprocess(img) for img in images], dim=0)
        return self._model(batch).cpu().numpy()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_class: type[nn.Module],
        **kwargs: object,
    ) -> InferenceWrapper:
        """Load model from a checkpoint and wrap it."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = model_class(**checkpoint.get("hparams", {}))
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model=model, **kwargs)
```

## Testing Abstractions

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

## When NOT to Abstract

### Do Not Abstract Single-Use Logic

```python
# BAD: Unnecessary abstraction for one-off logic
class ImagePreprocessor:
    def __init__(self, size):
        self.size = size

    def process(self, image):
        return cv2.resize(image, self.size)

# GOOD: Just use the function directly
resized = cv2.resize(image, (224, 224))
```

### Do Not Hide Critical Details

```python
# BAD: Hides GPU memory management
class AutoBatcher:
    def auto_batch(self, items):
        # Magically figures out batch size based on GPU memory
        # Developer has no idea what's happening
        ...

# GOOD: Be explicit about batch size
for batch in DataLoader(dataset, batch_size=32):
    ...
```

### Do Not Create Classes for Single Functions

```python
# BAD: A class with one method is just a function
class NMSProcessor:
    def __init__(self, threshold):
        self.threshold = threshold

    def process(self, boxes, scores):
        return nms(boxes, scores, self.threshold)

# GOOD: Use functools.partial or just pass the argument
from functools import partial

apply_nms = partial(nms, iou_threshold=0.5)
filtered = apply_nms(boxes, scores)
```

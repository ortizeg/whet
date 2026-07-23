# Wrapper Patterns

Scope: full worked wrappers around third-party APIs — context-managed video reading, validated image I/O, and a model inference wrapper.

## Contents

- [Pattern 1: VideoReader abstraction](#pattern-1-videoreader-abstraction)
- [Pattern 2: Image loading abstraction](#pattern-2-image-loading-abstraction)
- [Pattern 3: Model inference wrapper](#pattern-3-model-inference-wrapper)

## Pattern 1: VideoReader abstraction

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

## Pattern 2: Image loading abstraction

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

## Pattern 3: Model inference wrapper

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

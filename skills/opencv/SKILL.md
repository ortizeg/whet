---
name: opencv
description: >
  Use this skill when doing image or video processing with OpenCV — reading and writing
  video, camera capture, color-space conversions, drawing overlays, NumPy/PyTorch
  conversions, and building type-safe wrappers over OpenCV's C-style cv2 API. Reach for
  it any time you'd otherwise call cv2 directly for I/O or preprocessing, even if the user
  doesn't say "OpenCV" and just mentions frames, webcam, or resizing images. For plotting
  results see matplotlib; for scoring model quality see model-evaluation.
---

# OpenCV Skill

Clean, type-safe wrappers around OpenCV's C-style API: video reading/writing,
image abstractions, drawing utilities, color-space conversions, and camera
capture. OpenCV's Python API exposes the C++ interface almost directly — magic
integer properties, BGR-by-default channels, no type hints, and manual resource
management. These abstractions fix that with Pythonic, context-managed, type-safe
interfaces.

## VideoReader Abstraction

Define an ABC so you can swap implementations (OpenCV, FFmpeg, hardware decoders)
without changing application code.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


class VideoReaderBase(ABC):
    @abstractmethod
    def __init__(self, source: str | Path) -> None: ...

    @abstractmethod
    def read_frame(self) -> np.ndarray | None:
        """Read the next frame. Returns None at end of video."""
        ...

    @abstractmethod
    def seek(self, frame_number: int) -> None: ...

    @property
    @abstractmethod
    def metadata(self) -> VideoMetadata: ...

    @abstractmethod
    def __enter__(self) -> "VideoReaderBase": ...

    @abstractmethod
    def __exit__(self, *args) -> None: ...

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def read_frames(self, start: int = 0, count: int | None = None) -> list[np.ndarray]:
        self.seek(start)
        frames = []
        for frame in self:
            frames.append(frame)
            if count is not None and len(frames) >= count:
                break
        return frames
```

### OpenCV Implementation

Reads frames as RGB (converting from OpenCV's native BGR), validates the source,
and releases the capture on exit.

```python
import cv2


class OpenCVVideoReader(VideoReaderBase):
    def __init__(self, source: str | Path) -> None:
        self._path = Path(source)
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self._path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._metadata = VideoMetadata(
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=frame_count,
            duration_seconds=frame_count / max(fps, 1e-6),
            codec=self._decode_fourcc(int(self._cap.get(cv2.CAP_PROP_FOURCC))),
        )

    @staticmethod
    def _decode_fourcc(fourcc: int) -> str:
        return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def read_frame(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def seek(self, frame_number: int) -> None:
        if frame_number < 0 or frame_number >= self._metadata.frame_count:
            raise ValueError(f"Frame {frame_number} out of range [0, {self._metadata.frame_count})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def __enter__(self) -> "OpenCVVideoReader":
        return self

    def __exit__(self, *args) -> None:
        self._cap.release()

    def __del__(self) -> None:
        if hasattr(self, "_cap") and self._cap.isOpened():
            self._cap.release()
```

## Image Class Abstraction

Wraps a numpy array with explicit color-space tracking, validated construction,
and safe conversion methods — eliminating silent BGR/RGB bugs.

```python
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Self

import cv2
import numpy as np


class ColorSpace(Enum):
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"


class Image:
    """Type-safe image wrapper with color-space tracking."""

    def __init__(self, data: np.ndarray, color_space: ColorSpace = ColorSpace.RGB) -> None:
        if data.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")
        if data.ndim == 3 and data.shape[2] not in (1, 3, 4):
            raise ValueError(f"Expected 1, 3, or 4 channels, got {data.shape[2]}")
        if data.ndim == 2 and color_space != ColorSpace.GRAY:
            raise ValueError("2D array must use GRAY color space")
        self._data = data
        self._color_space = color_space

    @classmethod
    def from_file(cls, path: str | Path, color_space: ColorSpace = ColorSpace.RGB) -> Self:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        if color_space == ColorSpace.RGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cls(img, color_space=color_space)

    @classmethod
    def from_numpy(cls, array: np.ndarray, color_space: ColorSpace = ColorSpace.RGB) -> Self:
        return cls(array.copy(), color_space=color_space)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def color_space(self) -> ColorSpace:
        return self._color_space

    @property
    def height(self) -> int:
        return self._data.shape[0]

    @property
    def width(self) -> int:
        return self._data.shape[1]

    @property
    def channels(self) -> int:
        return self._data.shape[2] if self._data.ndim == 3 else 1

    def to_color_space(self, target: ColorSpace) -> Image:
        if target == self._color_space:
            return self
        conversion_map = {
            (ColorSpace.RGB, ColorSpace.BGR): cv2.COLOR_RGB2BGR,
            (ColorSpace.BGR, ColorSpace.RGB): cv2.COLOR_BGR2RGB,
            (ColorSpace.RGB, ColorSpace.GRAY): cv2.COLOR_RGB2GRAY,
            (ColorSpace.BGR, ColorSpace.GRAY): cv2.COLOR_BGR2GRAY,
            (ColorSpace.GRAY, ColorSpace.RGB): cv2.COLOR_GRAY2RGB,
            (ColorSpace.GRAY, ColorSpace.BGR): cv2.COLOR_GRAY2BGR,
            (ColorSpace.RGB, ColorSpace.HSV): cv2.COLOR_RGB2HSV,
            (ColorSpace.HSV, ColorSpace.RGB): cv2.COLOR_HSV2RGB,
            (ColorSpace.RGB, ColorSpace.LAB): cv2.COLOR_RGB2LAB,
            (ColorSpace.LAB, ColorSpace.RGB): cv2.COLOR_LAB2RGB,
        }
        key = (self._color_space, target)
        if key not in conversion_map:
            raise ValueError(f"Unsupported conversion: {self._color_space} -> {target}")
        return Image(cv2.cvtColor(self._data, conversion_map[key]), color_space=target)

    def resize(self, width: int, height: int, interpolation: int = cv2.INTER_LINEAR) -> Image:
        return Image(cv2.resize(self._data, (width, height), interpolation=interpolation),
                     color_space=self._color_space)

    def to_float32(self) -> Image:
        """Convert to float32 in [0, 1]."""
        if self._data.dtype == np.float32:
            return self
        return Image(self._data.astype(np.float32) / 255.0, color_space=self._color_space)

    def to_uint8(self) -> Image:
        """Convert to uint8 in [0, 255]."""
        if self._data.dtype == np.uint8:
            return self
        return Image((self._data * 255).clip(0, 255).astype(np.uint8), color_space=self._color_space)

    def to_tensor(self) -> "torch.Tensor":
        """Convert to a PyTorch (C, H, W) float32 tensor."""
        import torch
        img = self.to_float32().to_color_space(ColorSpace.RGB)
        return torch.from_numpy(img.data.transpose(2, 0, 1))

    def save(self, path: str | Path) -> None:
        cv2.imwrite(str(path), self.to_color_space(ColorSpace.BGR).data)
```

## Drawing Utilities

Named colors with automatic BGR conversion, plus box/keypoint/mask drawers. All
drawing functions `.copy()` the input since OpenCV mutates in place.

```python
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr(self) -> tuple[int, int, int]:
        return (self.b, self.g, self.r)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)


class Colors:
    RED = Color(255, 0, 0)
    GREEN = Color(0, 255, 0)
    BLUE = Color(0, 0, 255)
    YELLOW = Color(255, 255, 0)
    CYAN = Color(0, 255, 255)
    MAGENTA = Color(255, 0, 255)
    WHITE = Color(255, 255, 255)
    BLACK = Color(0, 0, 0)
    PALETTE = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA]

    @classmethod
    def for_class(cls, class_id: int) -> Color:
        """Consistent color per class ID."""
        return cls.PALETTE[class_id % len(cls.PALETTE)]


def draw_bounding_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],  # (x1, y1, x2, y2)
    label: str = "",
    color: Color = Colors.GREEN,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw a box with an optional filled label banner. Expects BGR input."""
    img = image.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color.bgr, thickness)
    if label:
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color.bgr, -1)
        cv2.putText(img, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return img


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,              # (N, 4) xyxy
    labels: list[str],
    scores: np.ndarray | None = None,
    class_ids: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw multiple boxes, coloring by class ID and appending scores to labels."""
    img = image.copy()
    for i in range(len(boxes)):
        color = Colors.for_class(class_ids[i] if class_ids is not None else i)
        text = f"{labels[i]} {scores[i]:.2f}" if scores is not None else labels[i]
        img = draw_bounding_box(img, tuple(boxes[i]), label=text, color=color, thickness=thickness)
    return img


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,          # (N, 2) or (N, 3) with confidence
    skeleton: list[tuple[int, int]] | None = None,
    color: Color = Colors.GREEN,
    radius: int = 4,
    thickness: int = 2,
) -> np.ndarray:
    """Draw keypoints and optional skeleton lines (drawn first, behind points)."""
    img = image.copy()
    if skeleton is not None:
        for start, end in skeleton:
            pt1 = tuple(keypoints[start, :2].astype(int))
            pt2 = tuple(keypoints[end, :2].astype(int))
            cv2.line(img, pt1, pt2, color.bgr, thickness)
    for kp in keypoints:
        conf = kp[2] if len(kp) > 2 else 1.0
        if conf > 0.5:
            cv2.circle(img, (int(kp[0]), int(kp[1])), radius, color.bgr, -1)
    return img


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,               # binary (H, W), values 0/1
    color: Color = Colors.GREEN,
    alpha: float = 0.4,
) -> np.ndarray:
    """Alpha-blend a binary mask over the image."""
    img = image.copy()
    overlay = img.copy()
    overlay[mask.astype(bool)] = color.bgr
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
```

## Camera Capture

Context-managed capture with configurable resolution/FPS; iterating yields frames
until a read fails.

```python
class Camera:
    def __init__(self, device_id: int = 0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self._device_id = device_id
        self._width, self._height, self._fps = width, height, fps
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._device_id}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Camera not opened. Use 'with' or call open().")
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __iter__(self):
        while True:
            try:
                yield self.read()
            except RuntimeError:
                break
```

## Video Writer

Context-managed writer expecting BGR frames.

```python
class VideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int, codec: str = "mp4v") -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to create video writer: {self._path}")
        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# Read RGB, process, convert back to BGR to write:
with OpenCVVideoReader("input.mp4") as reader:
    meta = reader.metadata
    with VideoWriter("output.mp4", meta.fps, meta.width, meta.height) as writer:
        for frame in reader:
            processed = process_frame(frame)
            writer.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
```

## NumPy / PyTorch Conversions

```python
import torch


def numpy_to_torch(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 tensor."""
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    return torch.from_numpy(image.transpose(2, 0, 1))


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """CHW float32 tensor -> HWC uint8."""
    arr = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return (arr * 255).clip(0, 255).astype(np.uint8)


def batch_to_numpy(batch: torch.Tensor) -> list[np.ndarray]:
    """BCHW tensor -> list of HWC uint8 arrays."""
    return [torch_to_numpy(batch[i]) for i in range(batch.shape[0])]
```

## Best Practices

1. **Always track color space** -- use the `Image` class or explicit naming (`frame_rgb`, `frame_bgr`) to avoid silent BGR/RGB confusion.
2. **Use context managers** -- wrap `VideoCapture`/`VideoWriter` in `with` blocks to guarantee release.
3. **Convert to RGB early** -- convert BGR->RGB right after reading; convert back only when writing or displaying with OpenCV.
4. **Abstract over backends** -- define ABCs so OpenCV can be swapped for FFmpeg, Decord, or hardware decoders.
5. **Validate inputs** -- check dtype, shape, and value range before processing.
6. **Copy before mutating** -- OpenCV drawing functions modify arrays in place.
7. **Use named constants** -- define color palettes and codec strings as module-level constants, not magic values.

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
interfaces. This page holds the everyday core; the deep dives below carry the full
implementations.

## The two rules that prevent most OpenCV bugs

1. **Track color space explicitly.** `cv2.imread` and `VideoCapture.read` return **BGR**.
   Convert to RGB immediately after reading and back to BGR only when writing or
   displaying with OpenCV.
2. **Always use context managers.** `VideoCapture` and `VideoWriter` hold OS handles and
   must be released, even on exceptions.

## The image load/save core

`cv2.imread` returns `None` for missing or corrupt files instead of raising, and returns
BGR. Check and convert immediately.

```python
from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB uint8, failing loudly on unreadable files."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(image_rgb: np.ndarray, path: str | Path) -> None:
    """Write an RGB uint8 image, converting back to OpenCV's BGR on the way out."""
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
```

## The video read/write core

```python
def process_video(input_path: str | Path, output_path: str | Path) -> None:
    """Read frames as RGB, process, write back as BGR — releasing both handles."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb = process_frame(frame_rgb)
            writer.write(cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
    finally:
        cap.release()
        writer.release()
```

Prefer the `OpenCVVideoReader` / `VideoWriter` classes in
`references/video-io.md` over raw `cv2.VideoCapture` in application code — they are
context-managed, iterable, and carry validated `VideoMetadata`.

## The safe-annotation core

OpenCV drawing functions mutate their input array in place, so always `.copy()` first,
and remember that color tuples are **BGR**.

```python
import numpy as np


def draw_box(image_bgr: np.ndarray, box: tuple[int, int, int, int], label: str) -> np.ndarray:
    """Draw one labelled box on a copy of a BGR image."""
    img = image_bgr.copy()
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BGR green
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
```

## Conventions

1. **Always track color space** -- use the `Image` class or explicit naming (`frame_rgb`, `frame_bgr`) to avoid silent BGR/RGB confusion.
2. **Use context managers** -- wrap `VideoCapture`/`VideoWriter` in `with` blocks to guarantee release.
3. **Convert to RGB early** -- convert BGR->RGB right after reading; convert back only when writing or displaying with OpenCV.
4. **Abstract over backends** -- define ABCs so OpenCV can be swapped for FFmpeg, Decord, or hardware decoders.
5. **Validate inputs** -- check dtype, shape, and value range before processing.
6. **Copy before mutating** -- OpenCV drawing functions modify arrays in place.
7. **Use named constants** -- define color palettes and codec strings as module-level constants, not magic values.

## Anti-patterns

- **Passing raw `cv2` arrays around untagged** — nothing records whether an array is RGB or BGR; wrap it in `Image` or name the variable for its color space.
- **Calling `cv2.rectangle`/`cv2.circle` on a caller's array** — it mutates in place and corrupts the caller's frame; copy first.
- **Leaking `VideoCapture` handles** — forgetting `release()` on an exception path exhausts device/file handles; use `with` or `try/finally`.
- **Magic `CAP_PROP_*` integers scattered through code** — read metadata once into a frozen dataclass.
- **Assuming `cv2.imread` succeeded** — it returns `None` for missing/corrupt files instead of raising; check explicitly.
- **Mixing uint8 and float32 ranges** — `[0, 255]` vs `[0, 1]` mismatches silently produce black or blown-out images.

## Deep dives

- `references/video-io.md` — read when building a video reader/writer, seeking to frames, iterating frames, or reading video metadata (fps, codec, frame count).
- `references/image-class-and-color-spaces.md` — read when you need a validated `Image` wrapper, color-space conversion tables (RGB/BGR/GRAY/HSV/LAB), resizing, or dtype conversion.
- `references/drawing-and-annotation.md` — read when drawing bounding boxes, detection overlays, keypoints/skeletons, or alpha-blended segmentation masks.
- `references/camera-capture.md` — read when capturing from a webcam or capture device with configurable resolution and FPS.
- `references/tensor-conversions.md` — read when converting between NumPy HWC images and PyTorch CHW/BCHW tensors.

# Drawing and Annotation

Scope: named color constants with automatic BGR conversion, plus drawers for bounding
boxes, detection sets, keypoints/skeletons, and alpha-blended masks.

## Contents

- [Colors](#colors)
- [Bounding Boxes](#bounding-boxes)
- [Detections](#detections)
- [Keypoints](#keypoints)
- [Mask Overlays](#mask-overlays)

## Drawing Utilities

Named colors with automatic BGR conversion, plus box/keypoint/mask drawers. All
drawing functions `.copy()` the input since OpenCV mutates in place.

### Colors

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
```

### Bounding Boxes

```python
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
```

### Detections

```python
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
```

### Keypoints

```python
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
```

### Mask Overlays

```python
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

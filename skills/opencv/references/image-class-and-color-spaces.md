# Image Class and Color Spaces

Scope: a type-safe `Image` wrapper around NumPy arrays with explicit color-space
tracking (RGB/BGR/GRAY/HSV/LAB), validated construction, conversion, resizing, dtype
conversion, and saving.

## Contents

- [ColorSpace Enum](#colorspace-enum)
- [Image Class Abstraction](#image-class-abstraction)
- [Usage](#usage)

## ColorSpace Enum

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
```

## Image Class Abstraction

Wraps a numpy array with explicit color-space tracking, validated construction,
and safe conversion methods — eliminating silent BGR/RGB bugs.

```python
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

## Usage

```python
img = Image.from_file("frame.png")                 # loaded and converted to RGB
small = img.resize(320, 320)                       # color space preserved
hsv = small.to_color_space(ColorSpace.HSV)         # explicit, validated conversion
tensor = small.to_tensor()                         # (C, H, W) float32 in [0, 1]
small.save("frame_small.png")                      # converted back to BGR for cv2.imwrite
```

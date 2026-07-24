# OpenCV Video I/O

Scope: reading and writing video files — a backend-agnostic reader ABC, the OpenCV
implementation, frame metadata, seeking, and a context-managed writer.

## Contents

- [VideoReader Abstraction](#videoreader-abstraction)
- [OpenCV Implementation](#opencv-implementation)
- [Video Writer](#video-writer)

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

## OpenCV Implementation

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

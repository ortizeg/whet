# Camera Capture

Scope: context-managed live capture from a webcam or capture device with configurable
resolution and FPS.

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

Note: `Camera.read()` returns frames in OpenCV's native **BGR** order. Convert with
`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` before handing frames to model preprocessing.

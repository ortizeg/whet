# Mocking External Dependencies

Testing code that depends on experiment trackers, cameras, GPUs, or other expensive resources.

## Patterns

Use mocking to test code that depends on external services, GPUs, or expensive resources.

```python
from unittest.mock import MagicMock, patch

import pytest


def test_wandb_logging_called():
    """Patch a module to verify metrics are logged without a real W&B run."""
    with patch("myproject.training.wandb") as mock_wandb:
        Trainer(use_wandb=True).log_metrics({"loss": 0.5}, step=100)
        mock_wandb.log.assert_called_once_with({"loss": 0.5}, step=100)


@pytest.fixture
def mock_camera():
    """Mock cv2.VideoCapture returning synthetic frames and properties."""
    camera = MagicMock()
    camera.read.return_value = (True, np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))
    camera.isOpened.return_value = True
    camera.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480, cv2.CAP_PROP_FPS: 30,
    }.get(prop, 0)
    return camera


def test_camera_processor_with_mock(mock_camera):
    with patch("cv2.VideoCapture", return_value=mock_camera):
        frame = CameraProcessor(camera_id=0).read_frame()
        assert frame.shape == (480, 640, 3)
```

## Notes

- Patch where the name is *used*, not where it is defined — `patch("myproject.training.wandb")`, not `patch("wandb")`.
- `MagicMock.get.side_effect` with a dict lookup is the clean way to emulate OpenCV's property-bag API.
- Mock external services and hardware; do not mock your own model or transform code — that tests the mock, not the behavior.

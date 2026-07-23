---
name: testing
description: >
  Use this skill when writing or fixing tests for ML/CV code with pytest — test structure,
  fixtures and conftest, parametrized tests, CV-specific assertions, tensor-shape
  validation, mocking external dependencies, performance tests, and coverage/CI. Reach
  for it any time code needs tests or a test is failing, even if the user just says "add
  tests" or "why is this test breaking". This is generic code-correctness testing; for
  scoring model quality with metrics like mAP, IoU, or calibration, see model-evaluation.
---

# Testing Skill

Pytest patterns for ML and computer vision projects: test structure, fixtures, parametrized tests, CV-specific strategies, mocking, performance testing, and CI. ML code fails silently — a model can train cleanly yet predict garbage from wrong preprocessing, label mapping, transposed dims, or broken augmentation. Tests catch these before they waste GPU hours.

## Coverage Requirements

- **Overall:** minimum 80% line coverage
- **New code:** at least 90%
- **Critical paths:** 100% — model forward pass, data loading, config validation
- **No skipped tests:** fix or delete them; a permanently skipped test is a lie

## Test Naming Convention

Pattern: `test_<what>_<condition>_<expected>` — the name should state the contract.

```python
def test_detector_empty_image_raises_error() -> None: ...
def test_config_negative_lr_raises_validation_error() -> None: ...
def test_model_forward_batch_returns_correct_shape() -> None: ...
```

## Test Structure

Three tiers: unit (fast, isolated), integration (component interactions), and e2e (full pipeline).

```
tests/
├── conftest.py              # repo-wide fixtures
├── unit/                    # test_model, test_transforms, test_dataset, test_metrics
├── integration/             # test_training_step, test_data_pipeline, test_inference
└── e2e/                     # test_train_eval, test_export
```
Each tier gets its own `conftest.py` for tier-scoped fixtures.

## The AAA Pattern

Every test follows Arrange-Act-Assert.

```python
def test_model_forward_pass():
    """Test that model produces correct output shape."""
    # Arrange
    model = ResNet50(num_classes=10)
    batch = torch.randn(4, 3, 224, 224)

    # Act
    output = model(batch)

    # Assert
    assert output.shape == (4, 10)
    assert not torch.isnan(output).any()
```

## Fixtures and conftest.py

Place shared fixtures in `conftest.py` at the appropriate level (repo-wide vs. per-tier).

### Top-Level conftest.py

```python
# tests/conftest.py
import numpy as np
import pytest
import torch


@pytest.fixture
def seed():
    """Set deterministic seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a synthetic RGB image (H, W, C) in uint8."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch(device) -> torch.Tensor:
    """Create a batch of normalized images (B, C, H, W)."""
    return torch.randn(4, 3, 224, 224, device=device)


@pytest.fixture
def sample_bboxes() -> np.ndarray:
    """Sample bounding boxes in xyxy format."""
    return np.array([[10, 20, 100, 150], [200, 50, 350, 300]], dtype=np.float32)


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory):
    """Small model with non-random weights for integration tests. Session-scoped for speed."""
    model = SmallTestModel(num_classes=5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(10):
        x = torch.randn(2, 3, 32, 32)
        y = torch.randint(0, 5, (2,))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    """Temporary ImageFolder-style dataset (train/val/test × classes)."""
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(5):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(cls_dir / f"{i:03d}.jpg"), img)
    return tmp_path
```

### Unit Test conftest.py

```python
# tests/unit/conftest.py
import pytest
from myproject.models import SmallResNet, EfficientNetTiny


@pytest.fixture(params=["small_resnet", "efficientnet_tiny"])
def model_factory(request):
    """Parametrized fixture that yields different model architectures."""
    models = {
        "small_resnet": lambda nc: SmallResNet(num_classes=nc),
        "efficientnet_tiny": lambda nc: EfficientNetTiny(num_classes=nc),
    }
    return models[request.param]
```

## Parametrized Tests

Use `@pytest.mark.parametrize` (stack decorators for a cross product) and `ids=` for readable case names.

```python
import pytest
import torch


@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("image_size", [224, 320, 640])
def test_model_handles_various_sizes(batch_size, image_size):
    """Model should handle different batch sizes and resolutions."""
    model = MyModel(num_classes=10)
    x = torch.randn(batch_size, 3, image_size, image_size)
    output = model(x)
    assert output.shape == (batch_size, 10)


@pytest.mark.parametrize(
    "bbox_format,expected",
    [
        ("xyxy", [10, 20, 110, 220]),
        ("xywh", [10, 20, 100, 200]),
        ("cxcywh", [60, 120, 100, 200]),
    ],
)
def test_bbox_conversion(bbox_format, expected):
    """Test bounding box format conversions."""
    result = convert_bbox([10, 20, 110, 220], from_format="xyxy", to_format=bbox_format)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    "num_classes,input_shape",
    [(10, (1, 3, 224, 224)), (1000, (1, 3, 384, 384))],
    ids=["10cls-single", "1000cls-highres"],
)
def test_classifier_output(num_classes, input_shape):
    """Test classifier with various class counts and inputs."""
    model = Classifier(num_classes=num_classes)
    x = torch.randn(*input_shape)
    out = model(x)
    assert out.shape == (input_shape[0], num_classes)
```

## CV-Specific Testing

### Testing with Synthetic Data

Never rely on real datasets in unit tests. Generate synthetic data that exercises the same code paths.

```python
import cv2
import numpy as np
import pytest


def make_synthetic_detection_sample(
    image_size: tuple[int, int] = (480, 640),
    num_objects: int = 5,
    num_classes: int = 10,
) -> dict:
    """Synthetic detection sample with image, boxes (xyxy), and labels."""
    h, w = image_size
    boxes = []
    for _ in range(num_objects):
        x1, y1 = np.random.randint(0, w - 50), np.random.randint(0, h - 50)
        x2 = np.random.randint(x1 + 10, min(x1 + 200, w))
        y2 = np.random.randint(y1 + 10, min(y1 + 200, h))
        boxes.append([x1, y1, x2, y2])
    return {
        "image": np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
        "boxes": np.array(boxes, dtype=np.float32),
        "labels": np.random.randint(0, num_classes, num_objects),
    }


def test_detection_dataset_returns_correct_types():
    """Verify dataset item structure and types."""
    sample = make_synthetic_detection_sample()
    assert sample["image"].dtype == np.uint8
    assert sample["boxes"].dtype == np.float32
    assert sample["boxes"].shape[1] == 4
    assert len(sample["labels"]) == len(sample["boxes"])
```

### Testing Augmentations

```python
import albumentations as A
import numpy as np
import pytest


@pytest.fixture
def augmentation_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.Resize(256, 256),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


def test_augmentation_preserves_bbox_count(augmentation_pipeline):
    """Augmentation should not drop bounding boxes."""
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    bboxes = [[10, 20, 100, 150], [200, 50, 350, 300]]
    labels = [0, 1]

    result = augmentation_pipeline(image=image, bboxes=bboxes, labels=labels)

    assert len(result["bboxes"]) == 2
    assert result["image"].shape == (256, 256, 3)


def test_horizontal_flip_mirrors_bboxes():
    """Horizontal flip should mirror bbox x-coordinates."""
    transform = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    bboxes = [[10, 20, 50, 80]]
    labels = [0]

    result = transform(image=image, bboxes=bboxes, labels=labels)
    # x1 becomes width - original_x2 = 200 - 50 = 150
    assert result["bboxes"][0][0] == pytest.approx(150, abs=1)
```

### Testing Video Processing

```python
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    """Write a 90-frame (3s @ 30fps) synthetic video."""
    video_path = tmp_path / "test_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    for i in range(90):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = int(320 + 200 * np.sin(2 * np.pi * i / 90))
        cv2.circle(frame, (x, 240), 30, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return video_path


def test_video_reader(synthetic_video):
    """VideoReader should report metadata and yield all frames."""
    reader = VideoReader(synthetic_video)
    assert reader.frame_count == 90
    assert reader.fps == pytest.approx(30.0, abs=0.1)
    assert reader.resolution == (640, 480)
    frames = list(reader)
    assert len(frames) == 90
    assert all(f.shape == (480, 640, 3) for f in frames)
```

## Mocking External Dependencies

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

## Performance Tests

Mark performance tests separately so they can be skipped in quick test runs.

```python
import time

import pytest


@pytest.mark.slow
def test_model_inference_speed(device):
    """Model inference should be under 50ms per image."""
    model = MyModel(num_classes=80).to(device).eval()
    x = torch.randn(1, 3, 640, 640, device=device)

    # Warmup
    for _ in range(10):
        model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_image = (elapsed / 100) * 1000
    assert ms_per_image < 50, f"Inference too slow: {ms_per_image:.1f}ms"
```

Always warm up and call `torch.cuda.synchronize()` around GPU timing. Apply the same pattern to `DataLoader` throughput tests.

## Edge Case Testing

```python
def test_empty_image():
    """Model should handle zero-size image gracefully."""
    model = MyModel(num_classes=10)
    with pytest.raises(ValueError, match="Image dimensions must be positive"):
        model(torch.randn(1, 3, 0, 0))


def test_no_detections():
    """Post-processing should return empty results when nothing detected."""
    results = postprocess(torch.zeros(1, 0, 6), confidence_threshold=0.5)
    assert len(results[0]["boxes"]) == 0
    assert len(results[0]["labels"]) == 0


def test_nan_in_loss():
    """Training should detect and raise on NaN loss."""
    with pytest.raises(RuntimeError, match="NaN"):
        Trainer(detect_nan=True).train_step(model, torch.tensor(float("nan")), optimizer)
```

Also cover: 1x1 images, all-same-class metrics, wrong dtypes, and maximum sizes.

## Coverage Configuration

Configure coverage in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: slow tests (deselect with -m 'not slow')", "gpu: requires GPU"]
addopts = ["--strict-markers", "-ra", "--tb=short"]

[tool.coverage.run]
source = ["src/myproject"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
```

Common invocations: `pytest --cov=src/myproject --cov-report=term-missing` (coverage), `-m "not slow"` (skip slow), `tests/unit/test_model.py` (one file), `-k "test_bbox"` (by pattern).

## CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src/myproject --cov-report=xml -m "not slow"
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

## Best Practices

1. **Test behavior, not implementation** — keeps refactoring safe.
2. **One assertion concept per test** — related asserts on the same concept are fine.
3. **Descriptive names** — `test_model_raises_on_wrong_input_channels`, not `test_model_error`.
4. **Keep unit tests in milliseconds** — mark slow ones with `@pytest.mark.slow`.
5. **Prefer fixtures over `setUp`/`tearDown`** — more composable.
6. **Test edge cases** — empty inputs, single elements, max sizes, NaN, wrong dtypes.
7. **Pin random seeds** via a seed fixture for determinism.
8. **Maintain 80%+ coverage** (`fail_under = 80`).
9. **Run tests in CI on every push** — never merge red.
10. **Assert data shapes explicitly** — shape mismatches are the most common ML bug.

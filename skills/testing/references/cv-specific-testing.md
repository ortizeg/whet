# CV-Specific Testing

Synthetic data generation, augmentation correctness, video processing, and edge cases for computer vision code.

## Contents

- [Testing with Synthetic Data](#testing-with-synthetic-data)
- [Testing Augmentations](#testing-augmentations)
- [Testing Video Processing](#testing-video-processing)
- [Edge Case Testing](#edge-case-testing)

## Testing with Synthetic Data

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

## Testing Augmentations

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

## Testing Video Processing

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

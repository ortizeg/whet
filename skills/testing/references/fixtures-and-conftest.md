# Fixtures and conftest.py

Shared pytest fixtures for ML/CV projects, and how to scope them across the test tiers.

## Contents

- [Test Structure](#test-structure)
- [Top-Level conftest.py](#top-level-conftestpy)
- [Unit Test conftest.py](#unit-test-conftestpy)

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

Place shared fixtures in `conftest.py` at the appropriate level (repo-wide vs. per-tier).

## Top-Level conftest.py

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

## Unit Test conftest.py

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

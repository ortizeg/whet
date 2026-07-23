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

Pytest patterns for ML and computer vision projects. ML code fails silently — a model can
train cleanly yet predict garbage from wrong preprocessing, label mapping, transposed dims,
or broken augmentation. Tests catch these before they waste GPU hours. This page is the
index; the detailed patterns live in `references/` and should be read only when needed.

## Test Layout and Naming

Three tiers, each with its own `conftest.py` for tier-scoped fixtures:

```
tests/
├── conftest.py              # repo-wide fixtures
├── unit/                    # fast, isolated: test_model, test_transforms, test_dataset
├── integration/             # component interactions: test_training_step, test_data_pipeline
└── e2e/                     # full pipeline: test_train_eval, test_export
```

Name tests `test_<what>_<condition>_<expected>` — the name should state the contract.

```python
def test_detector_empty_image_raises_error() -> None: ...
def test_config_negative_lr_raises_validation_error() -> None: ...
def test_model_forward_batch_returns_correct_shape() -> None: ...
```

## The Essential Core: AAA + a Fixture

Every test follows Arrange-Act-Assert; shared setup goes in a fixture in `conftest.py`.

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
def sample_batch() -> torch.Tensor:
    """Create a batch of normalized images (B, C, H, W)."""
    return torch.randn(4, 3, 224, 224)


# tests/unit/test_model.py
def test_model_forward_pass(sample_batch):
    """Test that model produces correct output shape."""
    # Arrange
    model = ResNet50(num_classes=10)

    # Act
    output = model(sample_batch)

    # Assert
    assert output.shape == (4, 10)
    assert not torch.isnan(output).any()
```

Shape and NaN assertions are the two highest-yield checks in ML testing — shape mismatches
are the most common bug, and NaNs are the most common silent one.

## Conventions

- **Test behavior, not implementation** — keeps refactoring safe.
- **One assertion concept per test** — related asserts on the same concept are fine.
- **Descriptive names** — `test_model_raises_on_wrong_input_channels`, not `test_model_error`.
- **Keep unit tests in milliseconds** — mark slow ones with `@pytest.mark.slow` and deselect with `-m "not slow"`.
- **Prefer fixtures over `setUp`/`tearDown`** — more composable.
- **Never use real datasets in unit tests** — generate synthetic data that exercises the same code paths.
- **Test edge cases** — empty inputs, single elements, max sizes, NaN, wrong dtypes.
- **Pin random seeds** via a seed fixture for determinism.
- **Assert data shapes explicitly** — shape mismatches are the most common ML bug.
- **Maintain 80%+ coverage** (`fail_under = 80`); 90% on new code, 100% on model forward, data loading, and config validation.
- **No skipped tests** — fix or delete them; a permanently skipped test is a lie.
- **Run tests in CI on every push** — never merge red.

## Deep dives

- `references/fixtures-and-conftest.md` — read when setting up a test suite or writing shared fixtures: the three-tier layout plus repo-wide and per-tier `conftest.py` examples (seed, device, images, bboxes, session-scoped trained model, temp dataset dir).
- `references/parametrized-tests.md` — read when one test body should cover many shapes, formats, or configurations with `@pytest.mark.parametrize` and readable `ids=`.
- `references/cv-specific-testing.md` — read when testing CV code specifically: synthetic detection samples, augmentation and bbox-mirroring correctness, synthetic video fixtures, and edge cases.
- `references/mocking.md` — read when code under test touches W&B, a camera, a GPU, or any other external or expensive resource.
- `references/performance-tests.md` — read when asserting inference latency or dataloader throughput, and for correct GPU timing (warmup + `torch.cuda.synchronize()`).
- `references/ci-integration.md` — read when configuring coverage and markers in `pyproject.toml` or wiring the GitHub Actions test workflow.

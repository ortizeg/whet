# Performance Tests

Timing inference and data loading correctly, and keeping slow tests out of the fast feedback loop.

## Inference Speed Test

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

## Notes

- Without warmup, the first iterations measure CUDA context creation and kernel autotuning, not the model.
- Without `torch.cuda.synchronize()`, CUDA calls are asynchronous and the timer measures queue submission rather than execution.
- Declare the `slow` marker in `pyproject.toml` so `--strict-markers` accepts it, then deselect with `-m "not slow"` in the fast CI job.
- Absolute latency thresholds are machine-dependent; keep them generous enough to avoid flakes, and treat them as regression guards rather than benchmarks.

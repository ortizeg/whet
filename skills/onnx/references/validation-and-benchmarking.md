# ONNX Validation and Benchmarking

Scope: proving an exported ONNX model matches its PyTorch source, and measuring ONNX
Runtime latency, percentiles, and throughput.

## Validation

Always validate that the ONNX model produces the same outputs as the original PyTorch model.

```python
import torch
import numpy as np
import onnxruntime as ort

def validate_onnx_export(
    pytorch_model: torch.nn.Module,
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    atol: float = 1e-5,
    rtol: float = 1e-5,
    num_tests: int = 5,
) -> bool:
    """Validate ONNX model matches PyTorch model outputs."""
    pytorch_model.eval()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    all_passed = True
    for _ in range(num_tests):
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]
        try:
            np.testing.assert_allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)
        except AssertionError:
            all_passed = False

    return all_passed

assert validate_onnx_export(model, "model.onnx"), "ONNX validation failed"
```

## Performance Benchmarking

```python
import time
import numpy as np
import onnxruntime as ort

def benchmark_onnx(
    model_path: str,
    input_shape: tuple[int, ...],
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict[str, float]:
    """Benchmark ONNX Runtime inference performance."""
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "throughput_fps": 1000.0 / float(np.mean(times)),
    }

results = benchmark_onnx("model.onnx", (1, 3, 640, 640))
```

To compute the ONNX speedup, run the same warmup/timed loop against the PyTorch model
(calling `torch.cuda.synchronize()` after each forward pass on GPU) and divide the mean
latencies.

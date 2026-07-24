# Benchmarking TensorRT

Measuring throughput and latency, and quantifying the speedup TensorRT gives over the plain CUDA execution provider.

## trtexec Benchmarking

```bash
# Reports throughput, latency percentiles, GPU compute time, and host latency
trtexec --onnx=model.onnx --fp16 --iterations=1000 --warmUp=500 --avgRuns=100
```

`--warmUp` is measured in milliseconds of warm-up before timing starts. Add `--dumpProfile`
to get per-layer timings and find the slow layers worth optimizing.

## ONNX Runtime CUDA vs TensorRT

```python
import time

import numpy as np
import onnxruntime as ort

from loguru import logger


def compare_backends(
    onnx_path: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    num_warmup: int = 50,
    num_runs: int = 500,
) -> None:
    """Compare ONNX Runtime CUDA vs TensorRT execution providers."""
    dummy = np.random.randn(*input_shape).astype(np.float32)

    results = {}
    for provider_name, providers in [
        ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("TensorRT", [
            ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]),
    ]:
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name

        # Warmup
        for _ in range(num_warmup):
            session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append((time.perf_counter() - start) * 1000)

        results[provider_name] = {
            "mean_ms": np.mean(times),
            "p99_ms": np.percentile(times, 99),
            "fps": 1000.0 / np.mean(times),
        }

        logger.info(
            "{}: {:.2f} ms mean, {:.2f} ms p99, {:.0f} FPS",
            provider_name,
            results[provider_name]["mean_ms"],
            results[provider_name]["p99_ms"],
            results[provider_name]["fps"],
        )

    speedup = results["CUDA"]["mean_ms"] / results["TensorRT"]["mean_ms"]
    logger.info("TensorRT speedup over CUDA: {:.2f}x", speedup)
```

## Measurement Rules

- **Never skip warmup.** The first inferences include JIT compilation, kernel selection, and
  memory allocation. Fifty warmup runs is a reasonable floor; TensorRT's first call may also
  build or load an engine.
- **Report p99, not just the mean.** Serving SLAs are set on tail latency, and TensorRT's
  advantage is often larger at the tail.
- **Benchmark at production batch size and resolution.** A batch-1 speedup does not predict
  batch-32 throughput.
- **Benchmark on the target GPU.** Engines are architecture-specific and so are the numbers.
- **Validate accuracy alongside speed.** Compare outputs against the ONNX/PyTorch reference —
  a fast engine that changed the predictions is not a win, especially with INT8.
- Expect roughly 2–6x over ONNX Runtime CUDA from kernel auto-tuning, layer fusion, and
  reduced precision; treat anything far outside that as a measurement bug.

---
name: onnx
description: >
  Use this skill when exporting a PyTorch model to ONNX or running inference with ONNX
  Runtime — configuring dynamic axes, optimizing and slimming the graph, graph surgery,
  validating ONNX outputs against PyTorch, quantization, and selecting execution
  providers. Reach for it any time a model needs to leave PyTorch for portable or
  optimized cross-platform inference, even if the user just says "export the model" or
  "run it without torch". For squeezing maximum NVIDIA-GPU performance out of the ONNX
  file, see tensorrt.
---

# ONNX Model Export and Inference

Export trained PyTorch models to ONNX for optimized inference with ONNX Runtime
(typically 2-10x faster, hardware-agnostic, no PyTorch dependency at deploy time).
This page holds the export/inference core you need almost every time; the deep dives
below cover providers, quantization, validation, and serving.

## The export core

Always call `model.eval()` before export, pass `dynamic_axes` for batch size at minimum,
slim the graph, then validate with `onnx.checker.check_model()`. Use `opset_version=17`+
for modern architectures.

```python
import onnx
import onnxslim
import torch

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    "model_raw.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"},
    },
)

onnx.save(onnxslim.slim(onnx.load("model_raw.onnx")), "model.onnx")
onnx.checker.check_model(onnx.load("model.onnx"))
```

Required export order: `torch.onnx.export()` → `onnxslim.slim()` →
`onnx.checker.check_model()` → (optional) quantize → (optional) benchmark.
Never deploy raw exported models, and never quantize before slimming.

## Dynamic axes at a glance

Dynamic axes let the model accept variable-size inputs at runtime — essential for
production where batch size and image size vary.

```python
# Image classification: variable batch size
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

# Object detection: variable batch size and image size
dynamic_axes = {
    "input": {0: "batch_size", 2: "height", 3: "width"},
    "output": {0: "batch_size"},
}

# Sequence models: variable batch and sequence length
dynamic_axes = {
    "input": {0: "batch_size", 1: "sequence_length"},
    "output": {0: "batch_size", 1: "sequence_length"},
}
```

## The inference core

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: np.random.randn(1, 3, 640, 640).astype(np.float32)})
```

## Conventions

1. **Always run `onnxslim.slim()`** after export — it removes redundant ops and reduces model size.
2. **Always validate** the exported model against PyTorch outputs.
3. **Use dynamic axes** for batch size at minimum.
4. **Set `opset_version=17`** or higher for modern architectures.
5. **Benchmark before and after** to confirm speedup.
6. **Use graph optimization** (`ORT_ENABLE_ALL`) in production.
7. **Consider quantization** for CPU deployment (2-4x speedup).
8. **Profile with ONNX Runtime** profiling tools to find bottlenecks.
9. **Wrap ONNX inference** in a Pydantic-validated class for type safety.
10. **Store ONNX models** in object storage or an artifact registry, not in Git.

## Anti-patterns

- **Missing dynamic axes** — without them the model only accepts the exact export shape; always pass `dynamic_axes` for batch size at minimum.
- **Unsupported ops** — some PyTorch ops don't export. Prefer tensor forms, e.g. `torch.where(cond, torch.ones_like(x), torch.zeros_like(x))` over scalar args.
- **Data type mismatch** — ONNX Runtime expects `float32`; cast inputs with `.astype(np.float32)`, not `float64`.
- **Forgetting `model.eval()`** — training-mode dropout/batchnorm produce different outputs; always call before export.
- **Opset too low** — use `opset_version=17`+ for modern architectures.
- **Quantizing before slimming** — always slim first, then quantize.
- **Creating a session per request** — build it once at startup and reuse it.

## Deep dives

- `references/export-and-dynamic-axes.md` — read when writing a reusable/validated export function, inspecting an ONNX model's input-output specs, or exporting a model with multiple inputs or outputs.
- `references/execution-providers-and-sessions.md` — read when configuring GPU/TensorRT providers, tuning session options, or wrapping ONNX Runtime in a typed inference class.
- `references/optimization-and-quantization.md` — read when slimming the graph, enabling ORT graph optimization, or applying INT8 dynamic/static quantization or FP16 conversion.
- `references/validation-and-benchmarking.md` — read when checking ONNX outputs match PyTorch, or measuring latency, percentiles, and throughput.
- `references/serving.md` — read when putting an ONNX model behind a FastAPI endpoint with a lifespan-managed session.

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

Export trained PyTorch models to ONNX for optimized inference with ONNX Runtime (typically 2-10x faster, hardware-agnostic, no PyTorch dependency at deploy time).

## Exporting PyTorch Models to ONNX

Always call `model.eval()` before export, run `onnx.checker.check_model()` to validate, and use `opset_version=17`+ for modern architectures.

### Export with Pydantic Configuration

```python
import torch
import onnx
from pydantic import BaseModel, Field

class ONNXExportConfig(BaseModel):
    """ONNX export configuration."""
    opset_version: int = Field(ge=11, default=17)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    input_names: list[str] = Field(default_factory=lambda: ["input"])
    output_names: list[str] = Field(default_factory=lambda: ["output"])

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    config: ONNXExportConfig,
) -> None:
    """Export PyTorch model to ONNX with validated configuration."""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=config.opset_version,
        input_names=config.input_names,
        output_names=config.output_names,
        dynamic_axes=config.dynamic_axes,
    )
    onnx.checker.check_model(onnx.load(output_path))

config = ONNXExportConfig(
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"},
    },
)
export_to_onnx(model, torch.randn(1, 3, 640, 640), "model.onnx", config)
```

## Dynamic Axes Configuration

Dynamic axes allow the model to accept variable-size inputs at runtime. This is essential for production deployment where batch sizes, image heights, and widths may vary.

### Common Configurations

```python
# Image classification: variable batch size
dynamic_axes = {
    "input": {0: "batch_size"},
    "output": {0: "batch_size"},
}

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

## Input and Output Specifications

### Inspecting Model Inputs and Outputs

```python
import onnx

def inspect_model(model_path: str) -> None:
    """Print input and output specifications of an ONNX model."""
    model = onnx.load(model_path)

    print("Inputs:")
    for inp in model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  {inp.name}: {shape} ({dtype})")

    print("\nOutputs:")
    for out in model.graph.output:
        shape = [dim.dim_value or dim.dim_param for dim in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  {out.name}: {shape} ({dtype})")

inspect_model("model.onnx")
```

### Multiple Inputs and Outputs

```python
import torch

class MultiInputModel(torch.nn.Module):
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        enhanced = self.metadata_encoder(metadata)
        boxes = self.detection_head(features, enhanced)
        scores = self.score_head(features, enhanced)
        return boxes, scores

# Export with multiple inputs/outputs
dummy_image = torch.randn(1, 3, 640, 640)
dummy_metadata = torch.randn(1, 10)

torch.onnx.export(
    model,
    (dummy_image, dummy_metadata),
    "multi_input_model.onnx",
    opset_version=17,
    input_names=["image", "metadata"],
    output_names=["boxes", "scores"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "metadata": {0: "batch_size"},
        "boxes": {0: "batch_size"},
        "scores": {0: "batch_size"},
    },
)
```

## ONNX Runtime Inference

### Basic Inference

```python
import numpy as np
import onnxruntime as ort

class ONNXInferenceSession:
    """ONNX Runtime inference wrapper."""

    def __init__(self, model_path: str) -> None:
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on a single input."""
        result = self.session.run(None, {self.input_name: input_array})
        return result[0]

    def predict_multi_output(self, input_array: np.ndarray) -> list[np.ndarray]:
        """Run inference returning all outputs."""
        return self.session.run(None, {self.input_name: input_array})

session = ONNXInferenceSession("model.onnx")
output = session.predict(np.random.randn(1, 3, 640, 640).astype(np.float32))
```

### Configuring Execution Providers

```python
import onnxruntime as ort

# GPU inference with fallback to CPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# GPU with specific options
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4 GB
        "cudnn_conv_algo_search": "EXHAUSTIVE",
    }),
    "CPUExecutionProvider",
]

# TensorRT for maximum GPU performance
providers = [
    ("TensorrtExecutionProvider", {
        "device_id": 0,
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
        "trt_fp16_enable": True,
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

session = ort.InferenceSession("model.onnx", providers=providers)
```

### Session Options

```python
import onnxruntime as ort

# Configure session options
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 4
session_options.enable_cpu_mem_arena = True
session_options.enable_mem_pattern = True
session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)
```

## Optimization

### OnnxSlim (Required)

OnnxSlim is a required post-export step: it reduces operators, removes redundant nodes, and folds constants — smaller/faster models without accuracy loss. Always slim after export, before quantization or deployment (`pixi add onnxslim`).

#### Python API

```python
import onnx
import onnxslim


def export_and_slim(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
) -> None:
    """Export PyTorch model to ONNX and slim it."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    raw_path = output_path.replace(".onnx", "_raw.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        raw_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        },
    )

    slimmed_model = onnxslim.slim(onnx.load(raw_path))
    onnx.save(slimmed_model, output_path)
    onnx.checker.check_model(onnx.load(output_path))
```

CLI equivalent: `onnxslim model_raw.onnx model.onnx`

Required export order: `torch.onnx.export()` → `onnxslim.slim()` → `onnx.checker.check_model()` → (optional) quantize → (optional) benchmark. Never deploy raw exported models or quantize before slimming.

### ORT Graph Optimization

ONNX Runtime also optimizes at session creation (complementary to OnnxSlim — use both). Set `session_options.graph_optimization_level = ORT_ENABLE_ALL` and `session_options.optimized_model_filepath = "model_optimized.onnx"` to persist the optimized graph.

### Quantization

Quantization reduces model size and increases inference speed by converting weights from FP32 to INT8:

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader

# Dynamic quantization (no calibration data needed)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
)

# Static quantization (requires calibration data)
class ImageCalibrationReader(CalibrationDataReader):
    """Provides calibration data for static quantization."""

    def __init__(self, calibration_images: list[np.ndarray]) -> None:
        self.images = iter(calibration_images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image = next(self.images)
            return {"input": image}
        except StopIteration:
            return None

calibration_reader = ImageCalibrationReader(calibration_images)

quantize_static(
    model_input="model.onnx",
    model_output="model_int8_static.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
)
```

### FP16 Conversion

```python
import onnx
from onnxconverter_common import float16

model_fp16 = float16.convert_float_to_float16(onnx.load("model.onnx"))
onnx.save(model_fp16, "model_fp16.onnx")
```

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

## Serving with ONNX Runtime and FastAPI

Load the session once in a `lifespan` handler; keep it in module scope for reuse across requests.

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]
    inference_time_ms: float = Field(ge=0)

# Global session
session: ort.InferenceSession | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    global session
    session = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    yield
    session = None

app = FastAPI(title="ONNX Detection API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Run object detection on an uploaded image."""
    import time
    import cv2

    # Read and preprocess image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    input_tensor = preprocess(image)  # Your preprocessing function

    # Run inference
    start = time.perf_counter()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.perf_counter() - start) * 1000

    # Parse outputs
    boxes, scores, labels = postprocess(outputs)  # Your postprocessing function

    return PredictionResponse(
        boxes=boxes.tolist(),
        scores=scores.tolist(),
        labels=labels.tolist(),
        inference_time_ms=inference_time,
    )
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

To compute the ONNX speedup, run the same warmup/timed loop against the PyTorch model (calling `torch.cuda.synchronize()` after each forward pass on GPU) and divide the mean latencies.

## Common Pitfalls

- **Missing dynamic axes** — without them the model only accepts the exact export shape; always pass `dynamic_axes` for batch size at minimum.
- **Unsupported ops** — some PyTorch ops don't export. Prefer tensor forms, e.g. `torch.where(cond, torch.ones_like(x), torch.zeros_like(x))` over scalar args.
- **Data type mismatch** — ONNX Runtime expects `float32`; cast inputs with `.astype(np.float32)`, not `float64`.
- **Forgetting `model.eval()`** — training-mode dropout/batchnorm produce different outputs; always call before export.
- **Opset too low** — use `opset_version=17`+ for modern architectures.

## Best Practices

1. **Always run `onnxslim.slim()`** after export — it removes redundant ops and reduces model size.
2. **Always validate** the exported model against PyTorch outputs.
3. **Use dynamic axes** for batch size at minimum.
4. **Set `opset_version=17`** or higher for modern architectures.
5. **Benchmark before and after** to confirm speedup.
6. **Use graph optimization** (`ORT_ENABLE_ALL`) in production.
7. **Consider quantization** for CPU deployment (2-4x speedup).
8. **Profile with ONNX Runtime** profiling tools to find bottlenecks.
9. **Wrap ONNX inference** in a Pydantic-validated class for type safety.
10. **Store ONNX models** as DVC-tracked artifacts, not in Git.

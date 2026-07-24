# ONNX Runtime Execution Providers and Sessions

Scope: creating and configuring `ort.InferenceSession` — typed inference wrappers,
execution provider selection (CPU/CUDA/TensorRT), and session option tuning.

## Contents

- [Basic Inference Wrapper](#basic-inference-wrapper)
- [Configuring Execution Providers](#configuring-execution-providers)
- [Session Options](#session-options)

## Basic Inference Wrapper

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

## Configuring Execution Providers

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

## Session Options

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

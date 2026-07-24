# TensorRT Inference at Runtime

Two ways to execute a TensorRT engine: the raw Python runtime with manual CUDA buffers, and the ONNX Runtime TensorRT execution provider (recommended for most projects).

## Contents

- [Raw TensorRT Runtime](#raw-tensorrt-runtime)
- [ONNX Runtime with TensorRT Backend](#onnx-runtime-with-tensorrt-backend)
- [Which to Use](#which-to-use)

## Raw TensorRT Runtime

Deserialize the engine, allocate device buffers once, then copy in/out per call. Use this only
when you need maximum control over memory and streams.

```python
from __future__ import annotations

import numpy as np
import tensorrt as trt
from cuda import cudart


class TensorRTInference:
    """TensorRT inference session."""

    def __init__(self, engine_path: str) -> None:
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate device memory
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate input/output GPU buffers."""
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.bindings: list[int] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = np.prod(shape) * np.dtype(dtype).itemsize

            # Allocate device memory
            err, device_mem = cudart.cudaMalloc(size)
            binding = {"name": name, "dtype": dtype, "shape": shape, "device": device_mem}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

            self.context.set_tensor_address(name, device_mem)

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        cudart.cudaMemcpy(
            self.inputs[0]["device"], input_array.ctypes.data, input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        self.context.execute_async_v3(0)
        output = np.empty(self.outputs[0]["shape"], dtype=self.outputs[0]["dtype"])
        cudart.cudaMemcpy(
            output.ctypes.data, self.outputs[0]["device"], output.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        return output

    def __del__(self) -> None:
        for buf in self.inputs + self.outputs:
            cudart.cudaFree(buf["device"])
```

Notes:

- Buffers are allocated **once** in `_allocate_buffers` and reused; allocating per call would
  dominate the latency you came here to save.
- `set_tensor_address` binds each named IO tensor to its device pointer; with TensorRT 10's
  name-based API there is no positional bindings list to maintain.
- `__del__` frees the CUDA allocations. Without it, a long-running service leaks device memory
  every time a session is replaced.
- Input arrays must be contiguous and of the exact dtype the engine expects (usually
  `np.float32`), otherwise `ctypes.data` copies garbage.
- With dynamic shapes, call `context.set_input_shape(name, shape)` before `execute_async_v3`.

## ONNX Runtime with TensorRT Backend

Recommended for most projects: TensorRT performance without managing raw CUDA memory. ONNX
Runtime handles engine building/caching and falls back to CUDA or CPU for unsupported layers.

```python
import onnxruntime as ort


def create_tensorrt_session(
    onnx_path: str,
    fp16: bool = True,
    max_workspace_size: int = 4 * 1024 * 1024 * 1024,
) -> ort.InferenceSession:
    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 0,
                "trt_max_workspace_size": max_workspace_size,
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache/",
            },
        ),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    return ort.InferenceSession(onnx_path, providers=providers)
```

- Provider order is the fallback chain: TensorRT first, then CUDA, then CPU. Unsupported
  subgraphs fall through automatically instead of failing the whole model.
- `trt_engine_cache_enable` + `trt_engine_cache_path` persist built engines across process
  restarts. Without it, every start pays the multi-minute build cost.
- The cache directory is GPU-architecture-specific — mount it per node, do not bake it into a
  portable image.

## Which to Use

Default to the ONNX Runtime TensorRT EP. Drop to the raw runtime only when you need custom
CUDA stream management, zero-copy input from another GPU pipeline, or per-layer control that
the EP does not expose.

# Building TensorRT Engines

Converting a slimmed ONNX model into a `.engine` file, with `trtexec` on the CLI or the `tensorrt.Builder` Python API, plus the Pydantic config models that drive both.

## Contents

- [Installation and CUDA Compatibility](#installation-and-cuda-compatibility)
- [trtexec (CLI)](#trtexec-cli)
- [Python Builder API](#python-builder-api)
- [Pydantic Configuration](#pydantic-configuration)

## Installation and CUDA Compatibility

```bash
pixi add tensorrt   # requires a matching CUDA toolkit
python -c "import tensorrt; print(tensorrt.__version__)"
```

CUDA compatibility: TensorRT 10.x → CUDA 12.x / cuDNN 9.x; TensorRT 8.6 → CUDA 11.8 or 12.x / cuDNN 8.9.

## trtexec (CLI)

`trtexec` is the fastest way to convert an ONNX model. It ships with TensorRT.

```bash
# Convert ONNX to TensorRT engine (FP32)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine

# FP16 precision (recommended default for GPU inference)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_fp16.engine \
    --fp16

# INT8 precision (requires calibration data)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model_int8.engine \
    --int8 \
    --calib=calibration_cache.bin
```

## Python Builder API

Use the Python API when you need programmatic control — conditional precision flags, custom
workspace sizing, or engine building inside an application's startup path.

```python
from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from loguru import logger


def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    max_batch_size: int = 1,
    max_workspace_size_gb: int = 4,
) -> None:
    """Build a TensorRT engine from an ONNX model."""
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX model
    onnx_path = Path(onnx_path)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TensorRT ONNX parse error: {}", parser.get_error(i))
            msg = f"Failed to parse ONNX model: {onnx_path}"
            raise RuntimeError(msg)

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size_gb * (1 << 30))

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("FP16 enabled")

    # Build engine
    logger.info("Building TensorRT engine from {} (this may take several minutes)", onnx_path.name)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        msg = "Failed to build TensorRT engine"
        raise RuntimeError(msg)

    # Save engine
    engine_path = Path(engine_path)
    engine_path.write_bytes(serialized_engine)
    logger.info("Engine saved to {} ({:.1f} MB)", engine_path, engine_path.stat().st_size / 1e6)
```

Notes:

- Always guard the FP16 flag with `builder.platform_has_fast_fp16` — enabling it on hardware
  without fast FP16 gains nothing and can cost accuracy.
- `build_serialized_network` returns `None` on failure; raise rather than writing an empty file.
- Building takes minutes. Cache the result and rebuild only when the ONNX model changes.
- Set the workspace pool generously (start at 4 GB). Too small a workspace prevents TensorRT
  from auto-tuning the fastest kernels.

## Pydantic Configuration

```python
from pydantic import BaseModel, Field


class TensorRTBuildConfig(BaseModel, frozen=True):
    """TensorRT engine build configuration."""

    onnx_path: str = Field(description="Path to slimmed ONNX model")
    engine_path: str = Field(description="Output engine path")
    fp16: bool = Field(default=True, description="Enable FP16 precision")
    int8: bool = Field(default=False, description="Enable INT8 precision")
    max_workspace_gb: int = Field(default=4, ge=1, le=32)
    max_batch_size: int = Field(default=1, ge=1)
    min_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    opt_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    max_shape: tuple[int, ...] = Field(default=(1, 3, 640, 640))
    calibration_cache: str | None = Field(default=None, description="INT8 calibration cache path")


class TensorRTInferenceConfig(BaseModel, frozen=True):
    engine_path: str = Field(description="Path to compiled engine")
    device_id: int = Field(default=0, ge=0)
```

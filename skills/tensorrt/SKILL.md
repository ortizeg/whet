---
name: tensorrt
description: >
  Use this skill when maximizing inference throughput and latency on NVIDIA GPUs by
  converting ONNX models to TensorRT engines — FP16/INT8 precision modes, dynamic shapes,
  engine building with trtexec or the Python API, INT8 calibration, benchmarking, and
  Triton Inference Server deployment. Reach for it any time NVIDIA-GPU inference must be
  squeezed to the limit, even if the user just says "make inference faster on the GPU".
  Assumes an ONNX model already exists — see onnx to produce one first.
---

# TensorRT Skill

Maximize inference performance on NVIDIA GPUs by converting ONNX models to TensorRT engines (typically 2-6x faster than ONNX Runtime CUDA via kernel auto-tuning, layer fusion, and FP16/INT8 precision). Requires the ONNX skill — TensorRT does not consume PyTorch directly.

Pipeline: `torch.onnx.export()` → `onnxslim.slim()` (ONNX skill) → `trtexec`/`tensorrt.Builder` (this skill) → `.engine`. Always slim before conversion; redundant ops removed early produce a better engine. Engines are GPU-architecture-specific (an A100 engine won't run on a T4) and take minutes to build — use TensorRT when deploying to known NVIDIA hardware for lowest latency.

## Essential Core

Two commands cover most of the work: convert with `trtexec --fp16`, or skip explicit engine
management entirely and let ONNX Runtime's TensorRT execution provider build and cache it.

```bash
pixi add tensorrt   # requires a matching CUDA toolkit

# FP16 conversion — the default for GPU inference
trtexec --onnx=model.onnx --saveEngine=model_fp16.engine --fp16

# Same, but accepting a range of batch sizes (tuned for the opt shape)
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640

# Sanity benchmark on the target GPU
trtexec --onnx=model.onnx --fp16 --iterations=1000 --warmUp=500 --avgRuns=100
```

Recommended runtime for most projects — TensorRT performance without managing CUDA buffers,
with automatic engine caching and fallback to CUDA/CPU for unsupported layers:

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

Provider order is the fallback chain: unsupported subgraphs drop to CUDA, then CPU, instead of
failing the model. `trt_engine_cache_enable` persists built engines across process restarts —
without it every start pays the multi-minute build cost.

## Decision Guide

| Situation | Path |
| --- | --- |
| Standard serving, want TensorRT speed with least effort | ONNX Runtime TensorRT EP (above) |
| Fixed input shape, ship a prebuilt artifact | `trtexec --fp16 --saveEngine` |
| Variable batch or resolution | Optimization profile — `references/dynamic-shapes.md` |
| Need the last few percent of latency | Raw `tensorrt.Builder` + runtime with manual CUDA buffers |
| Latency still too high at FP16 | INT8 with a real calibration set, then re-validate accuracy |

## Conventions

1. **Export to ONNX first** — run the ONNX skill pipeline (export → slim → validate) before conversion.
2. **Default to FP16** — halves memory with negligible accuracy loss on most models.
3. **Prefer the ONNX Runtime TensorRT EP** — it manages engine building/caching automatically; use the raw API only for maximum control.
4. **Cache built engines** by GPU architecture; rebuild only when the model changes.
5. **Set realistic dynamic profiles** — `opt_shape` should match your most common batch/resolution.
6. **Validate accuracy after conversion** against ONNX/PyTorch outputs.
7. **Profile with `trtexec --dumpProfile`** to find slow layers.
8. **Pin TensorRT and CUDA versions** in the Dockerfile and README.
9. **Use INT8 only with calibration data** — uncalibrated INT8 loses significant accuracy.
10. **Start the workspace pool at 4 GB** — TensorRT needs room for kernel auto-tuning.

CUDA compatibility: TensorRT 10.x → CUDA 12.x / cuDNN 9.x; TensorRT 8.6 → CUDA 11.8 or 12.x / cuDNN 8.9.

## Anti-Patterns

- ❌ Exporting PyTorch straight to TensorRT without ONNX + `onnxslim.slim()` first — slimming removes ops that confuse the optimizer.
- ❌ Assuming engines are portable — they are tied to the GPU architecture and TensorRT version.
- ❌ Using INT8 without calibration data.
- ❌ Setting `max_workspace_size` too low — TensorRT needs workspace for kernel auto-tuning; start with 4 GB.
- ❌ Skipping benchmark warmup — the first inferences include JIT compilation and allocation overhead.
- ❌ Using TensorRT on CPUs or non-NVIDIA GPUs — use ONNX Runtime for cross-platform deployment.
- ❌ Allocating CUDA buffers per inference call in the raw runtime — allocate once and reuse.

## Deep dives

- `references/engine-building.md` — read when converting ONNX to `.engine`, whether with `trtexec` or `tensorrt.Builder`, and for the Pydantic build/inference config models.
- `references/precision-and-calibration.md` — read when choosing FP32/FP16/INT8 or implementing an INT8 calibrator and cache.
- `references/dynamic-shapes.md` — read when the engine must accept variable batch sizes or input resolutions.
- `references/inference-runtime.md` — read when executing an engine: raw TensorRT runtime with CUDA buffers, or the ONNX Runtime TensorRT execution provider.
- `references/benchmarking.md` — read when measuring latency/throughput or comparing TensorRT against the CUDA execution provider.
- `references/docker-deployment.md` — read when containerizing a TensorRT service or deciding how to handle GPU-architecture-specific engines.

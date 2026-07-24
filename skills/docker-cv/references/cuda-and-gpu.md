# CUDA and GPU Containers

Scope: choosing a CUDA base image, pinning CUDA/driver versions, and the GPU-specific
runtime problems (shared memory, device visibility) that break CV containers.

## Base Image Selection

| Use Case | Base Image | Size |
|----------|-----------|------|
| Training (GPU) | `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` | ~3.5 GB |
| Training (CPU) | `python:3.11-slim` | ~150 MB |
| Inference (GPU) | `nvidia/cuda:12.4.1-runtime-ubuntu22.04` | ~2.8 GB |
| Inference (CPU) | `python:3.11-slim` | ~150 MB |
| Development | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` | ~5.2 GB |

Rules of thumb:

- `runtime` variants ship the CUDA runtime only; use them unless you compile CUDA kernels.
- `devel` variants ship `nvcc` and headers — needed to build custom ops, ~1.7 GB larger.
- `cudnn` variants add cuDNN, which PyTorch and TensorFlow both need for GPU training.
- For pre-built framework stacks, `nvcr.io/nvidia/pytorch:<yy.mm>-py3` is the NVIDIA NGC
  alternative to assembling the stack yourself.

## CUDA Version Mismatch

Always pin CUDA versions and document driver requirements:

```dockerfile
# Pin specific CUDA version
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Document minimum driver version in README
# Requires NVIDIA driver >= 550.54.15
```

The container supplies the CUDA runtime; the host supplies the driver. A container built
on CUDA 12.4 will fail on a host whose driver is too old, so the minimum driver version
belongs in the README next to the run instructions.

## Running with GPUs

```bash
# All GPUs
docker run --gpus all myproject:train

# Specific devices
docker run --gpus '"device=0,1"' myproject:train

# Verify the container can see the GPU
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If `nvidia-smi` works on the host but not in the container, the NVIDIA Container Toolkit
is missing or the daemon was not restarted after installing it.

## DataLoader Crashes (Shared Memory)

If `DataLoader` with `num_workers > 0` crashes with shared memory errors:

```yaml
# Increase shared memory size
services:
  train:
    shm_size: "8gb"  # Default is 64MB, way too small
```

The `docker run` equivalent is `--shm-size=8g`. This is the single most common cause of
`DataLoader worker (pid ...) is killed by signal: Bus error` in containerized training.

## Multi-Platform Builds

```bash
# Build for both AMD64 and ARM64
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag myproject:latest \
    --push .
```

CUDA base images are AMD64-only; multi-platform builds apply to CPU inference images
(for example ARM64 edge devices), not GPU training images.

# Deploying TensorRT in Containers

Packaging a TensorRT inference service, and handling the fact that engines are tied to a specific GPU architecture and TensorRT version.

## Dockerfile

```dockerfile
# Use NVIDIA's TensorRT container as base
FROM nvcr.io/nvidia/tensorrt:24.08-py3

WORKDIR /app

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

COPY pixi.toml pixi.lock ./
RUN pixi install

COPY models/ models/
COPY src/ src/

# Build the engine at startup (GPU-specific) or COPY a pre-built engine for the target GPU
ENTRYPOINT ["pixi", "run", "python", "-m", "my_project.serve"]
```

## Engine Portability

Engines are GPU-architecture-specific — build at container startup or ship separate images per
GPU target. An engine built on an A100 will not run on a T4, and an engine built against
TensorRT 8.6 will not load under TensorRT 10.

Three workable strategies:

1. **Build at startup.** Ship the ONNX model in the image and build the engine on first boot,
   writing it to a persistent volume. Costs minutes of startup time once per node; the
   readiness probe should stay red until the build finishes.
2. **Ship the ONNX model and use the ONNX Runtime TensorRT EP** with
   `trt_engine_cache_path` pointing at a mounted volume. Same effect as (1) but the runtime
   manages the cache for you — this is the lowest-friction option.
3. **Pre-build per target and bake in.** One image tag per GPU architecture, each with its
   `.engine` copied in. Fastest startup, most images to maintain.

## Operational Rules

- **Pin TensorRT and CUDA versions** in the Dockerfile and README. The base image tag
  (`24.08-py3`) fixes both — do not float it.
- **Never treat `.engine` files as build artifacts you can promote across environments** unless
  the GPU model and TensorRT version are identical.
- **Cache built engines by GPU architecture**, keyed on model hash, and rebuild only when the
  ONNX model changes.
- The container needs the NVIDIA container runtime and a GPU allocation — see the kubernetes
  skill for `nvidia.com/gpu` resource requests and node selectors.
- Do not run TensorRT on CPUs or non-NVIDIA GPUs; use ONNX Runtime for cross-platform
  deployment targets.

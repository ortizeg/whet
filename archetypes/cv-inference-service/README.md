# CV Inference Service Archetype

A production-ready project template for deploying trained computer vision models as REST APIs using FastAPI and ONNX Runtime. This archetype provides everything needed to serve models behind a scalable, containerized HTTP interface with health checks, request validation, structured logging, and GPU acceleration support.

## Purpose

The CV Inference Service archetype bridges the gap between a trained model checkpoint and a deployable prediction service. In most ML teams, the transition from a Jupyter notebook or training script to a production API is where projects stall. This archetype eliminates that friction by providing a fully structured FastAPI application with ONNX Runtime inference, Docker packaging with optional CUDA support, and production-grade observability hooks.

The service is designed around ONNX Runtime as the inference backend, which decouples the serving infrastructure from the training framework. Whether you trained your model in PyTorch, TensorFlow, or JAX, as long as you can export to ONNX format, this service can serve it. ONNX Runtime also provides significant latency improvements over native framework inference through graph optimization and hardware-specific acceleration.

## Use Cases

- **Real-time inference APIs** -- Serve object detection, classification, or segmentation models behind low-latency HTTP endpoints for web and mobile applications.
- **Batch processing services** -- Accept batches of images in a single request for high-throughput offline processing pipelines.
- **Model A/B testing** -- Run multiple model versions simultaneously behind different endpoints to compare performance in production.
- **Edge deployment** -- Build minimal Docker images that run on edge devices or IoT gateways with CPU-only inference.
- **Microservice architectures** -- Deploy as one component in a larger system, communicating via REST or gRPC with upstream and downstream services.
- **Model gateway** -- Front multiple specialized models behind a unified API that routes requests based on input characteristics.

## Directory Structure

This is exactly what `whet init` scaffolds -- nothing here is aspirational.

```
{{project_slug}}/
├── .env.example                     # Environment variable template
├── .gitignore
├── Dockerfile                       # pixi-based production image
├── README.md
├── pixi.toml                        # Environment + tasks (canonical)
├── pyproject.toml                   # Package metadata, ruff/mypy/pytest config
├── models/
│   └── .gitkeep                     # ONNX model storage (git-ignored artifacts)
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── py.typed
│   ├── app.py                       # FastAPI app, routes, lifespan model loading
│   ├── config.py                    # Settings via pydantic-settings
│   ├── schemas.py                   # Pydantic request/response models
│   └── inference/
│       ├── __init__.py
│       └── engine.py                # ONNX Runtime session + pre/postprocessing
└── tests/
    ├── __init__.py
    └── test_app.py                  # Green without any .onnx artifact
```

Grow it from there: split `schemas.py` into a `schemas/` package, add
`routes/`, `middleware/`, `k8s/`, or a GPU Dockerfile when you actually need
them. The recommended skills (`kubernetes`, `github-actions`, `tensorrt`) cover
those additions.

## Key Features

- **FastAPI** with async request handling for high concurrency under I/O-bound workloads.
- **ONNX Runtime** for framework-agnostic inference, with CUDA selected automatically when `DEVICE=cuda` and `CUDAExecutionProvider` is available.
- **Pydantic v2** request/response schemas with automatic OpenAPI documentation generation.
- **Guarded model loading** -- the model is loaded once in the FastAPI lifespan hook, and a missing `.onnx` file logs a warning and marks the service not-ready instead of crashing it, so a freshly scaffolded project runs and tests green out of the box.
- **Health and readiness probes** compatible with Kubernetes liveness and readiness checks (`/ready` answers 503 until a model is loaded).
- **Loguru logging** configured from `LOG_LEVEL` at startup.
- **pixi-based Docker image** running as a non-root user, with a `HEALTHCHECK` wired to `/health`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe -- returns 200 whenever the process is running |
| GET | `/ready` | Readiness probe -- 200 once the model is loaded, 503 before that |
| POST | `/predict` | Single image prediction (JSON body, base64-encoded image) |
| POST | `/predict/upload` | Single image prediction (multipart file upload) |
| GET | `/docs` | Interactive Swagger UI documentation |

## Configuration and Environment Variables

Configuration is managed through `pydantic-settings` in `src/{{package_name}}/config.py`, which reads from environment variables with an optional `.env` file fallback. All settings are validated at startup, and `.env.example` ships with the template.

| Variable | Description | Default |
|---|---|---|
| `APP_HOST` | Bind address | `0.0.0.0` |
| `APP_PORT` | Listen port | `8000` |
| `MODEL_NAME` | Logical model name for API responses | `default` |
| `MODEL_PATH` | Path to the ONNX model file | `models/model.onnx` |
| `DEVICE` | Inference device (`cpu` or `cuda`) | `cpu` |
| `INPUT_WIDTH` | Model input width in pixels | `640` |
| `INPUT_HEIGHT` | Model input height in pixels | `640` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections | `0.5` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `CORS_ORIGINS` | Comma-separated allowed CORS origins | `*` |

## Dependencies

Managed by `pixi.toml` (conda) and `pyproject.toml` (PyPI metadata).

```toml
[dependencies]
python = "3.11.*"
numpy = ">=1.26"
pillow = ">=10.0"

[pypi-dependencies]
fastapi = ">=0.110"
uvicorn = { version = ">=0.27", extras = ["standard"] }
pydantic = ">=2.6"
pydantic-settings = ">=2.2"
loguru = ">=0.7"
onnxruntime = ">=1.17"
python-multipart = ">=0.0.9"
```

## Usage

### Local Development

```bash
# Install dependencies
pixi install

# Copy environment template
cp .env.example .env

# Place your ONNX model in models/ (optional -- the service starts without it)
cp /path/to/your/model.onnx models/model.onnx

# Start the development server with hot reload
uvicorn {{package_name}}.app:app --reload --port 8000

# ...or via the pixi task
pixi run dev
```

Add dependencies with `pixi add <pkg>` (conda) or `pixi add --pypi <pkg>`.

### Docker Deployment

```bash
# Build the image (bootstraps pixi, runs `pixi install`, runs as non-root)
docker build -t {{project_slug}}:latest .

# Run container
docker run -p 8000:8000 -v $(pwd)/models:/app/models {{project_slug}}:latest
```

For GPU serving, swap `onnxruntime` for `onnxruntime-gpu`, base the image on an
NVIDIA CUDA runtime image, set `DEVICE=cuda`, and run with `--gpus all`.

### Making Predictions

```bash
# Base64 JSON body
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$(base64 < test_image.jpg)\", \"confidence_threshold\": 0.5}"

# Multipart upload
curl -X POST 'http://localhost:8000/predict/upload?confidence_threshold=0.5' \
  -F "file=@test_image.jpg"

# Check health and readiness
curl http://localhost:8000/health
curl -i http://localhost:8000/ready
```

### Quality Gates

```bash
pytest
ruff check .
mypy src/ --strict
```

## Customization Guide

### Adding Custom Preprocessing and Postprocessing

`src/{{package_name}}/inference/engine.py` ships a generic pipeline: resize ->
normalize (ImageNet mean/std) -> NCHW float32 -> `session.run` -> decode. The
decoder understands an `(N, >=6)` detection tensor (`x1, y1, x2, y2, score,
class_id`) and a 1-D class-logit vector; anything else logs a warning and
returns no detections.

Replace `OnnxInferenceEngine.preprocess` with the transforms your model was
exported with, and `OnnxInferenceEngine.postprocess` with your task-specific
decoding -- NMS and box-format conversion for detection, argmax and contour
extraction for segmentation, softmax and top-k label mapping for
classification. Set `labels` on `InferenceConfig` for human-readable classes.

### Adding New Endpoints

1. Define Pydantic request and response schemas in `src/{{package_name}}/schemas.py` (split it into a `schemas/` package once it grows).
2. Add the route to `app.py`, or create a `routes/` package and register the router.
3. Depend on the shared engine via the `EngineDep` annotated dependency so the model stays loaded once per process.
4. Add corresponding tests in `tests/`.

### Scaling Considerations

For CPU inference, scale horizontally with `uvicorn --workers N` or multiple container replicas behind a load balancer -- the service is stateless apart from the loaded session. For GPU inference, use one worker per GPU and scale by adding GPU nodes. The `kubernetes` skill covers Deployment, Service, and HPA manifests; wire the liveness probe to `/health` and the readiness probe to `/ready`.

### Switching Inference Backends

While ONNX Runtime is the default, the engine abstraction in `inference/engine.py` can be swapped for TensorRT, OpenVINO, or TorchScript by implementing the same interface. The rest of the application remains unchanged because preprocessing, postprocessing, and routing are decoupled from the inference backend.

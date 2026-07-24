# ${project_name}

${description}

A FastAPI service that serves a computer vision model with ONNX Runtime.

## Layout

```
${project_slug}/
├── Dockerfile                 # pixi-based production image
├── pixi.toml                  # environment + tasks (canonical)
├── pyproject.toml             # package metadata, ruff/mypy/pytest config
├── .env.example               # every supported environment variable
├── models/                    # drop your model.onnx here
├── src/${package_name}/
│   ├── app.py                 # FastAPI app, routes, lifespan model loading
│   ├── config.py              # pydantic-settings configuration
│   ├── schemas.py             # Pydantic request/response models
│   └── inference/
│       └── engine.py          # ONNX Runtime session, pre/postprocessing
└── tests/
    └── test_app.py            # runs green without any .onnx artifact
```

## Setup

```bash
pixi install
cp .env.example .env
```

Add a dependency with `pixi add <pkg>` (conda) or `pixi add --pypi <pkg>`.

Place your exported model where `MODEL_PATH` points:

```bash
cp /path/to/your/model.onnx models/model.onnx
```

The service starts **without** a model file — `/health` stays 200, while
`/ready` and `/predict` answer 503 until a model is loaded.

## Running

```bash
# Development server (hot reload)
uvicorn ${package_name}.app:app --reload --port 8000

# Production
uvicorn ${package_name}.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Via pixi tasks: `pixi run dev` / `pixi run serve`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe — 200 whenever the process is up |
| GET | `/ready` | Readiness probe — 503 until the ONNX model is loaded |
| POST | `/predict` | Inference on a base64-encoded image (JSON body) |
| POST | `/predict/upload` | Inference on a multipart image upload |
| GET | `/docs` | Swagger UI |

```bash
curl http://localhost:8000/health
curl -i http://localhost:8000/ready

curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"image_b64\": \"$$(base64 < test.jpg)\", \"confidence_threshold\": 0.5}"

curl -X POST 'http://localhost:8000/predict/upload?confidence_threshold=0.5' \
  -F 'file=@test.jpg'
```

## Configuration

Settings are read from environment variables (or a local `.env`) and validated
at startup by `src/${package_name}/config.py`. See `.env.example`.

| Variable | Description | Default |
|---|---|---|
| `APP_HOST` | Bind address | `0.0.0.0` |
| `APP_PORT` | Listen port | `8000` |
| `MODEL_NAME` | Logical model name in API responses | `default` |
| `MODEL_PATH` | Path to the ONNX model file | `models/model.onnx` |
| `DEVICE` | Inference device (`cpu` or `cuda`) | `cpu` |
| `INPUT_WIDTH` | Model input width in pixels | `640` |
| `INPUT_HEIGHT` | Model input height in pixels | `640` |
| `CONFIDENCE_THRESHOLD` | Default minimum detection confidence | `0.5` |
| `LOG_LEVEL` | Loguru log level | `INFO` |
| `CORS_ORIGINS` | Comma-separated allowed CORS origins | `*` |

## Adapting the engine to your model

`src/${package_name}/inference/engine.py` ships a generic pipeline:
resize → normalise (ImageNet mean/std) → NCHW float32 → `session.run` → decode.
The decoder understands two common export shapes — an `(N, >=6)` detection
tensor (`x1, y1, x2, y2, score, class_id`) and a 1-D class-logit vector.
Replace `preprocess`/`postprocess` with the transforms your model was exported
with, and set `labels` on `InferenceConfig` for human-readable class names.

CUDA is selected automatically when `DEVICE=cuda` *and* `CUDAExecutionProvider`
is available (install `onnxruntime-gpu`); otherwise the engine logs a warning
and falls back to CPU.

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy src/ --strict
```

Via pixi tasks: `pixi run test`, `pixi run lint`, `pixi run typecheck`, or
`pixi run quality` for all of them.

## Docker

```bash
docker build -t ${project_slug}:latest .
docker run -p 8000:8000 -v "$$(pwd)/models:/app/models" ${project_slug}:latest
```

The image bootstraps pixi and runs `pixi install`. For reproducible builds,
commit the generated `pixi.lock`, add it to the `COPY` line in the `Dockerfile`,
and switch to `pixi install --locked`.

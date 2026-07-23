---
name: fastapi
description: >
  Use this skill when building an HTTP API to serve an ML model — async prediction
  endpoints, Pydantic request/response schemas, dependency injection, middleware, CORS,
  background tasks, WebSocket streaming, health checks, and structured error handling.
  Reach for it any time you'd otherwise hand-roll a model-serving web service or expose
  inference over REST, even if the user just says "put this model behind an API". For a
  quick interactive demo UI instead of a production JSON API, see gradio.
---

# FastAPI Skill

Build FastAPI applications for serving ML models and CV pipelines. Use Pydantic models for all request/response schemas — never accept raw dicts. Use an application factory for testable configuration and clean startup/shutdown lifecycle.

## Application Factory

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field


class AppConfig(BaseModel, frozen=True):
    title: str = "ML Model API"
    version: str = "1.0.0"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    model_path: str = "models/best.onnx"
    max_batch_size: int = 32
    device: str = "cuda:0"


class ModelRegistry:
    """Holds loaded models for the application lifetime."""
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}

    async def load(self, config: AppConfig) -> None:
        logger.info("Loading model from {}", config.model_path)
        self.models["default"] = await _load_model(config.model_path)

    async def shutdown(self) -> None:
        self.models.clear()


model_registry = ModelRegistry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage model loading on startup and cleanup on shutdown."""
    config = AppConfig()
    await model_registry.load(config)
    yield
    await model_registry.shutdown()


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or AppConfig()

    app = FastAPI(
        title=config.title,
        version=config.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    from .routes import prediction, health
    app.include_router(health.router, tags=["health"])
    app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])

    return app
```

## Request and Response Models

Define explicit Pydantic models for every endpoint. Never use `dict` or `Any` in API signatures.

```python
from __future__ import annotations

import base64

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel, frozen=True):
    """Single image prediction request."""

    image_b64: str = Field(..., description="Base64-encoded image bytes")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=1000)

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        try:
            base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("Invalid base64 encoding") from exc
        return v


class Detection(BaseModel, frozen=True):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")


class PredictionResponse(BaseModel, frozen=True):
    detections: list[Detection]
    inference_time_ms: float
    model_version: str


class ErrorResponse(BaseModel, frozen=True):
    error: str
    detail: str | None = None
    request_id: str | None = None
```

## Prediction Endpoint with Dependency Injection

Inject the model and preprocessor via `Depends` — never load them inside the handler. For batch endpoints, accept a `list[PredictionRequest]` (cap with `Field(max_length=...)`) and loop this same logic.

```python
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends

from .schemas import PredictionRequest, PredictionResponse, Detection
from .dependencies import get_model, get_preprocessor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: Annotated[Model, Depends(get_model)],
    preprocessor: Annotated[Preprocessor, Depends(get_preprocessor)],
) -> PredictionResponse:
    """Run inference on a single image."""
    start = time.perf_counter()

    image = preprocessor.decode_and_preprocess(request.image_b64)
    raw_detections = await model.predict(image)

    detections = [
        Detection(label=d.label, confidence=d.confidence, bbox=d.bbox)
        for d in raw_detections
        if d.confidence >= request.confidence_threshold
    ][: request.max_detections]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
        model_version=model.version,
    )
```

### Health Check Endpoints

```python
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel, frozen=True):
    status: str
    model_loaded: bool
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness probe."""
    from .app import model_registry

    return HealthResponse(
        status="healthy",
        model_loaded=bool(model_registry.models),
        version="1.0.0",
    )


@router.get("/ready")
async def readiness() -> dict[str, bool]:
    """Kubernetes readiness probe — 503 until the model is loaded."""
    from .app import model_registry

    if not model_registry.models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}
```

## Dependency Injection

```python
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Load application configuration once."""
    return AppConfig()


async def get_model(
    config: AppConfig = Depends(get_app_config),
) -> Model:
    """Provide the loaded model instance."""
    from .app import model_registry

    model = model_registry.models.get("default")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


async def get_preprocessor(
    config: AppConfig = Depends(get_app_config),
) -> Preprocessor:
    """Provide the image preprocessor."""
    return Preprocessor(target_size=(640, 640), device=config.device)
```

## Middleware and Error Handling

### Request ID and Logging Middleware

```python
from __future__ import annotations

import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests with timing and request ID."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "{} {} → {} ({:.1f}ms) [{}]",
            request.method, request.url.path, response.status_code, elapsed, request_id,
        )
        response.headers["X-Request-ID"] = request_id
        return response
```

### Structured Exception Handlers

```python
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger


async def model_error_handler(request: Request, exc: ModelInferenceError) -> JSONResponse:
    logger.error("Inference error: {}", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "inference_error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ModelInferenceError, model_error_handler)
```

## WebSocket Streaming (Real-Time Video Inference)

```python
import base64

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


@router.websocket("/ws/stream")
async def stream_inference(websocket: WebSocket) -> None:
    """Stream video frames and return detections in real time."""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await websocket.receive_json()
            frame_b64 = data.get("frame")
            if not frame_b64:
                await websocket.send_json({"error": "Missing 'frame' field"})
                continue

            frame_bytes = base64.b64decode(frame_b64)
            detections = await run_inference(frame_bytes)

            await websocket.send_json({
                "detections": [d.model_dump() for d in detections],
                "frame_id": data.get("frame_id"),
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
```

## Background Tasks (Async Post-Processing)

Use `BackgroundTasks` to persist results or log after the response is sent, keeping request latency low.

```python
from fastapi import BackgroundTasks


async def save_prediction_to_db(request_id: str, detections: list[Detection]) -> None:
    await db.predictions.insert_one(
        {"request_id": request_id, "detections": [d.model_dump() for d in detections]}
    )


@router.post("/predict")
async def predict_with_logging(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    result = await run_prediction(request)
    background_tasks.add_task(save_prediction_to_db, request.state.request_id, result.detections)
    return result
```

## Testing (Async Test Client)

```python
from __future__ import annotations

import base64

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.fixture
async def client() -> AsyncClient:
    """Create async test client."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_predict_returns_detections(client: AsyncClient) -> None:
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    request_body = {
        "image_b64": base64.b64encode(image_bytes).decode(),
        "confidence_threshold": 0.5,
    }
    response = await client.post("/api/v1/predict", json=request_body)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "inference_time_ms" in data


@pytest.mark.anyio
async def test_predict_rejects_invalid_base64(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/predict",
        json={"image_b64": "not-valid-base64!!!"},
    )
    assert response.status_code == 422
```

## Docker Deployment

Production Dockerfile plus a Uvicorn runner for programmatic startup.

```dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"

COPY pixi.toml pixi.lock ./
RUN pixi install

COPY src/ ./src/

EXPOSE 8000

CMD ["pixi", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

For programmatic startup with the factory, use `uvicorn.run("app.main:create_app", factory=True, host="0.0.0.0", port=8000, workers=4, limit_concurrency=100, timeout_keep_alive=30)`.

## Anti-Patterns

- **Never use `dict` for request/response models** — always define Pydantic schemas with explicit fields and validators.
- **Never load models inside endpoint functions** — use the lifespan context manager for startup/shutdown lifecycle.
- **Never block the event loop with synchronous inference** — use `await` with async model wrappers or `run_in_executor` for CPU-bound operations.
- **Never hardcode CORS origins in production** — load from configuration.
- **Never return raw numpy arrays or tensors** — serialize to lists or base64 in the response model.
- **Never skip input validation** — use Pydantic field validators for base64, image dimensions, and parameter bounds.

## Integration with Other Skills

Pydantic (frozen models), Docker CV (multi-stage serving images), ONNX/TensorRT (load optimized models in the lifespan handler), Loguru (middleware/handler logging), Testing (httpx async client).

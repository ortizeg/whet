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

Build FastAPI applications for serving ML models and CV pipelines. Use Pydantic models for
all request/response schemas — never accept raw dicts. Load models once in the lifespan
handler and inject them into endpoints with `Depends`; use an application factory so the
app is configurable and testable.

## Essential Core: A Typed Prediction Endpoint

Define explicit Pydantic models for the request and response, then inject the model and
preprocessor. This is the shape of nearly every endpoint in a serving API.

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
```

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

## Conventions

- **Application factory.** `create_app(config) -> FastAPI` with a frozen `AppConfig`; start it
  with `uvicorn.run("app.main:create_app", factory=True, ...)`.
- **Lifespan owns model loading.** An `@asynccontextmanager` lifespan loads into a
  `ModelRegistry` at startup and clears it at shutdown.
- **Everything through `Depends`.** Config providers use `@lru_cache(maxsize=1)`; tests swap
  implementations with `app.dependency_overrides`.
- **`Annotated[T, Depends(...)]`** in handler signatures so MyPy and OpenAPI both see the type.
- **Declare `response_model`** on every route; return frozen Pydantic models, never dicts.
- **Structured errors.** One `ErrorResponse` shape (`error`, `detail`, `request_id`) for all
  failures, produced by registered exception handlers.
- **Request IDs.** Middleware assigns one, stores it on `request.state`, and echoes it in
  `X-Request-ID`.
- **Two health routes.** `/health` for liveness (cheap, model-independent), `/ready` for
  readiness (503 until the model is loaded).
- **Loguru brace formatting** in middleware and handlers, not f-strings.

## Anti-Patterns

- **Never use `dict` for request/response models** — always define Pydantic schemas with explicit fields and validators.
- **Never load models inside endpoint functions** — use the lifespan context manager for startup/shutdown lifecycle.
- **Never block the event loop with synchronous inference** — use `await` with async model wrappers or `run_in_executor` for CPU-bound operations.
- **Never hardcode CORS origins in production** — load from configuration.
- **Never return raw numpy arrays or tensors** — serialize to lists or base64 in the response model.
- **Never skip input validation** — use Pydantic field validators for base64, image dimensions, and parameter bounds.
- **Never rely on background tasks for work that must not be lost** — they die with the process; use a real queue.

## Integration with Other Skills

Pydantic (frozen models), Docker CV (multi-stage serving images), ONNX/TensorRT (load optimized models in the lifespan handler), Loguru (middleware/handler logging), Testing (httpx async client).

## Deep dives

- `references/application-factory-and-lifespan.md` — read when setting up the app object, `AppConfig`, `ModelRegistry`, or startup/shutdown model loading.
- `references/request-response-models.md` — read when designing schemas, adding validators, capping batch requests, or shaping error payloads.
- `references/dependency-injection.md` — read when wiring the model/preprocessor/config into endpoints or overriding dependencies in tests.
- `references/middleware-and-cors.md` — read when configuring CORS or adding request-ID/timing/logging middleware.
- `references/health-and-errors.md` — read when adding liveness/readiness probes or mapping domain exceptions to JSON error responses.
- `references/background-tasks.md` — read when deferring persistence or logging until after the response is sent.
- `references/websocket-streaming.md` — read when adding WebSocket or streaming responses for real-time video inference.
- `references/testing-and-deployment.md` — read when writing async httpx tests, building the Docker image, or tuning Uvicorn workers.

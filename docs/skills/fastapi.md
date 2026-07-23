# FastAPI

The FastAPI skill provides expert patterns for building ML model serving APIs with FastAPI, covering async endpoints, Pydantic request/response validation, dependency injection, middleware, and production deployment.

**Skill directory:** `skills/fastapi/`

## Purpose

FastAPI is the standard framework for serving ML models over HTTP. This skill encodes the best practices for structuring FastAPI applications in ML/CV contexts: application factory pattern with lifespan-managed model loading, typed request/response schemas with Pydantic validation, dependency injection for model and preprocessor instances, and production deployment with Uvicorn.

## When to Use

Use this skill whenever you need to:

- Serve trained models (ONNX, TensorRT, PyTorch) behind REST endpoints
- Build real-time WebSocket streaming for video inference pipelines
- Construct microservices wrapping ML inference
- Create APIs with health checks, batch endpoints, and structured error responses

## Key Patterns

### Application Factory with Lifespan

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Load models on startup, release on shutdown."""
    await model_registry.load(config)
    yield
    await model_registry.shutdown()


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    app = FastAPI(title=config.title, lifespan=lifespan)
    app.include_router(prediction.router, prefix="/api/v1")
    return app
```

### Pydantic Request/Response Schemas

```python
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel, frozen=True):
    image_b64: str = Field(..., description="Base64-encoded image")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("image_b64")
    @classmethod
    def validate_base64(cls, v: str) -> str:
        import base64
        base64.b64decode(v, validate=True)
        return v


class PredictionResponse(BaseModel, frozen=True):
    detections: list[Detection]
    inference_time_ms: float
    model_version: str
```

### Dependency Injection for Models

```python
from typing import Annotated
from fastapi import Depends


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: Annotated[Model, Depends(get_model)],
    preprocessor: Annotated[Preprocessor, Depends(get_preprocessor)],
) -> PredictionResponse:
    image = preprocessor.decode_and_preprocess(request.image_b64)
    detections = await model.predict(image)
    return PredictionResponse(detections=detections, ...)
```

## Anti-Patterns to Avoid

- Do not use `dict` for request/response schemas -- define explicit Pydantic models
- Do not load models inside endpoint functions -- use lifespan handlers
- Do not block the event loop with synchronous inference -- use async or `run_in_executor`
- Do not hardcode CORS origins in production -- load from configuration
- Do not return raw numpy arrays -- serialize to lists or base64

## Combines Well With

- **Pydantic** -- Frozen BaseModel patterns for all API schemas
- **ONNX / TensorRT** -- Optimized model formats loaded in lifespan
- **Docker CV** -- Production containerization for FastAPI services
- **Loguru** -- Structured logging in middleware and exception handlers
- **Testing** -- Async endpoint testing with httpx

## Full Reference

See [`skills/fastapi/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/fastapi/SKILL.md) for complete patterns including WebSocket streaming, background tasks, and middleware.

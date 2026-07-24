# Health Checks and Structured Error Handling

Liveness/readiness endpoints for orchestrators, and exception handlers that turn domain errors into a consistent JSON error shape.

## Health Check Endpoints

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

### Liveness vs Readiness

- **`/health` (liveness)** answers "is the process alive?" — it must stay cheap and must not depend on the model. If it fails, the orchestrator restarts the container.
- **`/ready` (readiness)** answers "can this replica serve traffic?" — it returns 503 while the model is still loading so the load balancer withholds traffic instead of returning errors. Model load can take minutes; pair this with a generous startup probe.

Neither endpoint should run inference or touch external services.

## Structured Exception Handlers

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

## Notes

- The handler body mirrors the `ErrorResponse` schema (`error`, `detail`, `request_id`) so every failure path returns the same shape clients already parse.
- `request.state.request_id` is set by the logging middleware; `getattr(..., None)` keeps the handler safe if the middleware is not installed.
- Register one handler per domain exception type (`ModelInferenceError`, `PreprocessingError`, ...) rather than catching broad exceptions inside endpoints. Handlers keep the happy path in the endpoint clean.
- Log at `error` level with the exception, but return only a safe message to the client — never leak stack traces or file paths in the response body.
- Call `register_exception_handlers(app)` from `create_app` after routers are included.

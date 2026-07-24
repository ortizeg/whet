# Application Factory and Lifespan

How to build a testable FastAPI app object: a frozen config model, a model registry, an async lifespan handler for startup/shutdown, and a `create_app()` factory that wires middleware and routers.

## Why a Factory

A module-level `app = FastAPI()` is hard to test — you cannot vary configuration per test. A `create_app(config)` factory lets tests build a fresh app with test settings, and lets Uvicorn start it with `factory=True`.

## Full Factory

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

## Notes

- The lifespan context manager is the **only** place a model should be loaded. Loading inside an endpoint blocks the first request and reloads per worker request.
- Everything before `yield` runs at startup; everything after runs at shutdown.
- Routers are imported inside `create_app` to avoid import cycles between the app module and route modules that depend on the registry.
- `AppConfig` is frozen so configuration cannot drift at runtime. Load real values from environment variables or a settings model rather than hardcoding CORS origins.

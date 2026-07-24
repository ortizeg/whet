# Dependency Injection

How to provide the model, preprocessor, and configuration to endpoints via `Depends` instead of loading them inside handlers.

## Dependencies Module

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

## Consuming Dependencies

Use `Annotated[T, Depends(provider)]` in the handler signature — this keeps the type visible to MyPy and to FastAPI's OpenAPI generation:

```python
from typing import Annotated

from fastapi import Depends


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: Annotated[Model, Depends(get_model)],
    preprocessor: Annotated[Preprocessor, Depends(get_preprocessor)],
) -> PredictionResponse:
    ...
```

## Rules

- **`@lru_cache(maxsize=1)` on config providers.** Configuration is read once per process; without the cache, every request re-parses environment variables.
- **Model providers read from the registry, never load.** `get_model` looks up the object that the lifespan handler already loaded. If it is missing, raise `HTTPException(503)` — that is a genuine "not ready" condition, not a server bug.
- **Dependencies compose.** `get_preprocessor` depends on `get_app_config`; FastAPI resolves the graph and caches each provider's result for the duration of a single request.
- **Override in tests.** `app.dependency_overrides[get_model] = lambda: FakeModel()` swaps in a stub without touching the production code path — this is the main reason to route everything through `Depends`.
- **Keep providers cheap.** Anything expensive (GPU allocation, file reads, network calls) belongs in the lifespan handler, with the provider returning the already-built object.

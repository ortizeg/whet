# Testing and Deployment

Async test client patterns for a model-serving API, plus the production Dockerfile and Uvicorn runner.

## Contents

- [Testing (Async Test Client)](#testing-async-test-client)
- [Docker Deployment](#docker-deployment)
- [Programmatic Uvicorn Startup](#programmatic-uvicorn-startup)

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

Notes:

- `ASGITransport` drives the app in-process — no network, no server to start.
- The factory is what makes this work: each test builds its own app instance.
- Test the rejection paths (422 on bad base64, 503 before the model loads) as deliberately as the happy path — validation is a feature of the API contract.
- Swap the real model with `app.dependency_overrides[get_model] = lambda: FakeModel()` to keep tests fast and GPU-free.

## Docker Deployment

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

Dependency files are copied before the source so that the `pixi install` layer stays cached across source-only changes.

## Programmatic Uvicorn Startup

For programmatic startup with the factory, use:

```python
uvicorn.run(
    "app.main:create_app",
    factory=True,
    host="0.0.0.0",
    port=8000,
    workers=4,
    limit_concurrency=100,
    timeout_keep_alive=30,
)
```

- `factory=True` tells Uvicorn the target is a callable returning an app, not an app object.
- Each worker is a separate process and therefore loads its own copy of the model — size `workers` against GPU memory, not CPU count.
- `limit_concurrency` sheds load with 503s instead of queueing unboundedly under a traffic spike.

# Request and Response Models

Every FastAPI endpoint in an ML service takes and returns an explicit Pydantic model — never `dict`, never `Any`.

## Full Schema Module

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

## Design Rules

- **Frozen models everywhere.** Request and response payloads are values, not mutable state.
- **Validate at the boundary.** Base64 decoding, image dimension bounds, and threshold ranges belong in `field_validator` / `Field` constraints — not in the handler body. FastAPI converts a `ValueError` in a validator into a 422 with a structured error body for free.
- **Bound every list.** `Field(min_length=..., max_length=...)` on `bbox`, and on any batch request list, prevents unbounded payloads.
- **Never return raw tensors or numpy arrays.** Serialize to Python lists (as `Detection.bbox` does) or base64 strings.
- **Declare `response_model`** on the route decorator so FastAPI both validates the outgoing payload and publishes an accurate OpenAPI schema.

## Batch Endpoints

For a batch endpoint, accept a `list[PredictionRequest]` capped with `Field(max_length=...)` on a wrapper model, and loop the same single-item logic:

```python
class BatchPredictionRequest(BaseModel, frozen=True):
    items: list[PredictionRequest] = Field(min_length=1, max_length=32)


class BatchPredictionResponse(BaseModel, frozen=True):
    results: list[PredictionResponse]
```

Cap `max_length` at the model's real batch capacity (`AppConfig.max_batch_size`) so an oversized request fails validation instead of exhausting GPU memory.

## Error Payloads

`ErrorResponse` is the single shape every failure returns. Register exception handlers that produce it (see the health-and-errors reference) so clients never have to parse two different error formats.

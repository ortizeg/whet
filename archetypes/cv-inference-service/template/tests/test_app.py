"""Tests for the inference service.

The whole suite runs **without any .onnx artifact present** — a freshly
scaffolded project must be green before a model exists.
"""

from __future__ import annotations

import base64
import io
from collections.abc import AsyncGenerator
from pathlib import Path

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from ${package_name}.app import app
from ${package_name}.inference.engine import (
    ImageDecodeError,
    InferenceConfig,
    OnnxInferenceEngine,
    decode_base64_image,
)


def _png_bytes(width: int = 32, height: int = 24) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """ASGI client that runs the real lifespan, model load included."""
    transport = ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=transport, base_url="http://test") as ac,
    ):
        yield ac


@pytest.fixture
def engine(tmp_path: Path) -> OnnxInferenceEngine:
    return OnnxInferenceEngine(
        InferenceConfig(model_path=tmp_path / "missing.onnx", input_width=8, input_height=8)
    )


# --------------------------------------------------------------------- routes


async def test_health_ok_without_model(client: AsyncClient) -> None:
    """The app must start and stay live when no model file exists."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


async def test_ready_reports_not_ready_without_model(client: AsyncClient) -> None:
    response = await client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


async def test_predict_returns_503_without_model(client: AsyncClient) -> None:
    payload = {"image_b64": base64.b64encode(_png_bytes()).decode(), "confidence_threshold": 0.5}
    response = await client.post("/predict", json=payload)
    assert response.status_code == 503


async def test_predict_upload_returns_503_without_model(client: AsyncClient) -> None:
    files = {"file": ("image.png", _png_bytes(), "image/png")}
    response = await client.post("/predict/upload", files=files)
    assert response.status_code == 503


async def test_predict_rejects_invalid_threshold(client: AsyncClient) -> None:
    payload = {"image_b64": "aGk=", "confidence_threshold": 5.0}
    response = await client.post("/predict", json=payload)
    assert response.status_code == 422


async def test_openapi_exposes_predict(client: AsyncClient) -> None:
    schema = (await client.get("/openapi.json")).json()
    assert "/predict" in schema["paths"]
    assert "PredictionResponse" in schema["components"]["schemas"]


# --------------------------------------------------------------------- engine


def test_load_missing_model_is_not_fatal(engine: OnnxInferenceEngine) -> None:
    assert engine.load() is False
    assert engine.is_ready is False
    assert engine.providers == []


def test_predict_without_model_raises(engine: OnnxInferenceEngine) -> None:
    with pytest.raises(RuntimeError):
        engine.predict(Image.new("RGB", (8, 8)))


def test_resolve_providers_defaults_to_cpu(engine: OnnxInferenceEngine) -> None:
    assert engine.resolve_providers() == ["CPUExecutionProvider"]


def test_preprocess_shape_and_dtype(engine: OnnxInferenceEngine) -> None:
    batch = engine.preprocess(Image.new("RGB", (64, 48), color=(255, 0, 0)))
    assert batch.shape == (1, 3, 8, 8)
    assert batch.dtype == np.float32


def test_postprocess_decodes_detection_rows(engine: OnnxInferenceEngine) -> None:
    raw = np.array([[1.0, 2.0, 3.0, 4.0, 0.9, 0.0], [1.0, 2.0, 3.0, 4.0, 0.1, 1.0]], np.float32)
    detections = engine.postprocess([raw], threshold=0.5)
    assert len(detections) == 1
    assert detections[0].label == "class_0"
    assert detections[0].bbox == [1.0, 2.0, 3.0, 4.0]


def test_postprocess_decodes_class_logits(engine: OnnxInferenceEngine) -> None:
    detections = engine.postprocess([np.array([0.1, 9.0, 0.2], np.float32)], threshold=0.5)
    assert len(detections) == 1
    assert detections[0].label == "class_1"


def test_postprocess_unknown_shape_is_empty(engine: OnnxInferenceEngine) -> None:
    assert engine.postprocess([np.zeros((2, 3, 4, 5), np.float32)], threshold=0.5) == []


def test_decode_base64_image_roundtrip() -> None:
    image = decode_base64_image(base64.b64encode(_png_bytes(16, 8)).decode())
    assert image.size == (16, 8)


def test_decode_base64_image_rejects_garbage() -> None:
    with pytest.raises(ImageDecodeError):
        decode_base64_image("not-base64!!")

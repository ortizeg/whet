"""FastAPI application for ${project_name}."""

from __future__ import annotations

import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL.Image import Image as PILImage

from . import __version__
from .config import Settings, get_settings
from .inference.engine import (
    ImageDecodeError,
    InferenceConfig,
    OnnxInferenceEngine,
    decode_base64_image,
    decode_image_bytes,
)
from .schemas import HealthResponse, PredictionRequest, PredictionResponse, ReadinessResponse


def build_engine(settings: Settings) -> OnnxInferenceEngine:
    """Create an engine from validated settings (performs no I/O)."""
    return OnnxInferenceEngine(
        InferenceConfig(
            model_path=settings.model_path,
            model_name=settings.model_name,
            device=settings.device,
            input_width=settings.input_width,
            input_height=settings.input_height,
            confidence_threshold=settings.confidence_threshold,
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the ONNX model once on startup and release it on shutdown."""
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)

    engine = build_engine(settings)
    app.state.settings = settings
    app.state.engine = engine

    # Guarded: a missing model must not stop the service from starting.
    if engine.load():
        logger.info("Service ready — model '{}' loaded", settings.model_name)
    else:
        logger.warning("Service started without a model; /predict will answer 503")

    yield

    engine.unload()
    logger.info("Shutting down")


app = FastAPI(
    title="${project_name}",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine(request: Request) -> OnnxInferenceEngine:
    """Return the engine created during startup."""
    engine = getattr(request.app.state, "engine", None)
    if not isinstance(engine, OnnxInferenceEngine):  # pragma: no cover - startup invariant
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="inference engine is not initialised",
        )
    return engine


EngineDep = Annotated[OnnxInferenceEngine, Depends(get_engine)]


def _require_ready(engine: OnnxInferenceEngine) -> None:
    if not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"no model loaded (expected at '{engine.config.model_path}')",
        )


async def _run(
    engine: OnnxInferenceEngine, image: PILImage, threshold: float
) -> PredictionResponse:
    detections, elapsed_ms = await run_in_threadpool(engine.predict, image, threshold)
    return PredictionResponse(detections=detections, inference_time_ms=elapsed_ms)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe — 200 while the process is up, with or without a model."""
    return HealthResponse(status="healthy", version=__version__)


@app.get("/ready", response_model=ReadinessResponse)
async def ready(engine: EngineDep, response: Response) -> ReadinessResponse:
    """Readiness probe — 503 until an ONNX model is loaded."""
    if not engine.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        ready=engine.is_ready,
        model_name=engine.config.model_name,
        model_path=str(engine.config.model_path),
        providers=engine.providers,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, engine: EngineDep) -> PredictionResponse:
    """Run inference on a base64-encoded image."""
    _require_ready(engine)
    try:
        image = decode_base64_image(payload.image_b64)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return await _run(engine, image, payload.confidence_threshold)


@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_upload(
    engine: EngineDep,
    file: Annotated[UploadFile, File(description="Image file to run inference on")],
    confidence_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.5,
) -> PredictionResponse:
    """Run inference on a multipart-uploaded image file."""
    _require_ready(engine)
    try:
        image = decode_image_bytes(await file.read())
    except ImageDecodeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return await _run(engine, image, confidence_threshold)

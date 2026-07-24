"""ONNX Runtime inference engine for ${project_name}.

The engine is deliberately small and typed end to end:

* :class:`InferenceConfig` — validated (Pydantic V2) description of the model.
* :class:`OnnxInferenceEngine` — loads the session, preprocesses an image, runs
  it, and decodes the raw tensors into :class:`~..schemas.Detection` objects.

A freshly scaffolded project has no ``.onnx`` file. Loading is therefore
*guarded*: a missing or unreadable model logs a warning and leaves the engine in
a not-ready state instead of crashing the service, so ``/health`` keeps working
and ``/ready`` correctly reports 503.

Replace :meth:`OnnxInferenceEngine.preprocess` and
:meth:`OnnxInferenceEngine.postprocess` with the transforms your own model was
exported with.
"""

from __future__ import annotations

import base64
import binascii
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from loguru import logger
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from ..schemas import Detection

FloatArray = NDArray[np.float32]

#: ImageNet normalisation — the most common export-time convention.
DEFAULT_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
DEFAULT_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class ImageDecodeError(ValueError):
    """Raised when request payload bytes are not a decodable image."""


class InferenceConfig(BaseModel, frozen=True):
    """Everything the engine needs to know about the model."""

    model_config = ConfigDict(frozen=True, protected_namespaces=())

    model_path: Path = Path("models/model.onnx")
    model_name: str = "default"
    device: str = Field(default="cpu", pattern="^(cpu|cuda)$")
    input_width: int = Field(default=640, gt=0)
    input_height: int = Field(default=640, gt=0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    mean: tuple[float, float, float] = DEFAULT_MEAN
    std: tuple[float, float, float] = DEFAULT_STD
    labels: tuple[str, ...] = ()


def decode_image_bytes(data: bytes) -> Image.Image:
    """Decode raw bytes into a PIL image, or raise :class:`ImageDecodeError`."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageDecodeError("payload is not a decodable image") from exc
    return image


def decode_base64_image(payload: str) -> Image.Image:
    """Decode a base64 (optionally data-URI prefixed) image string."""
    encoded = payload.split(",", 1)[-1] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError("image_b64 is not valid base64") from exc
    if not raw:
        raise ImageDecodeError("image_b64 is empty")
    return decode_image_bytes(raw)


class OnnxInferenceEngine:
    """Thin, typed wrapper around an ONNX Runtime session."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def is_ready(self) -> bool:
        """``True`` once a session has been successfully created."""
        return self._session is not None

    @property
    def providers(self) -> list[str]:
        """Execution providers actually in use (empty when not loaded)."""
        if self._session is None:
            return []
        providers: list[str] = list(self._session.get_providers())
        return providers

    def resolve_providers(self) -> list[str]:
        """Pick execution providers: CPU by default, CUDA when asked for and available."""
        available = set(ort.get_available_providers())
        if self.config.device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.warning(
                "device=cuda requested but CUDAExecutionProvider is unavailable; "
                "falling back to CPU (install onnxruntime-gpu for CUDA support)"
            )
        return ["CPUExecutionProvider"]

    def load(self) -> bool:
        """Load the ONNX session. Returns ``False`` instead of raising on failure."""
        path = self.config.model_path
        if not path.is_file():
            logger.warning(
                "ONNX model not found at '{}' — starting in a not-ready state. "
                "Drop a model there (or set MODEL_PATH) and restart.",
                path,
            )
            self._session = None
            return False

        try:
            session = ort.InferenceSession(str(path), providers=self.resolve_providers())
        except Exception as exc:  # noqa: BLE001 - never let a bad artifact kill startup
            logger.error("Failed to load ONNX model '{}': {}", path, exc)
            self._session = None
            return False

        self._session = session
        self._input_name = str(session.get_inputs()[0].name)
        logger.info(
            "Loaded ONNX model '{}' from '{}' (providers={}, input='{}')",
            self.config.model_name,
            path,
            session.get_providers(),
            self._input_name,
        )
        return True

    def unload(self) -> None:
        """Release the session."""
        if self._session is not None:
            logger.info("Releasing ONNX session for '{}'", self.config.model_name)
        self._session = None
        self._input_name = None

    # ------------------------------------------------------------- pre/post ops

    def preprocess(self, image: Image.Image) -> FloatArray:
        """Resize, normalise and convert an image to a NCHW float32 batch of 1."""
        resized = image.convert("RGB").resize(
            (self.config.input_width, self.config.input_height),
            Image.Resampling.BILINEAR,
        )
        scaled = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray(self.config.mean, dtype=np.float32)
        std = np.asarray(self.config.std, dtype=np.float32)
        normalised = ((scaled - mean) / std).astype(np.float32)
        chw = np.transpose(normalised, (2, 0, 1))
        batch: FloatArray = np.expand_dims(chw, axis=0).astype(np.float32)
        return batch

    def postprocess(self, outputs: list[Any], threshold: float) -> list[Detection]:
        """Decode raw model outputs into detections.

        Two common export shapes are handled out of the box; anything else logs a
        warning and returns no detections. Replace this with your model's decoder
        (NMS, anchor decoding, segmentation argmax, ...).
        """
        if not outputs:
            return []

        raw = np.squeeze(np.asarray(outputs[0], dtype=np.float32))

        # (N, >=6): [x1, y1, x2, y2, score, class_id] — typical exported detector.
        if raw.ndim == 2 and raw.shape[-1] >= 6:
            detections: list[Detection] = []
            for row in raw:
                score = float(row[4])
                if score < threshold:
                    continue
                detections.append(
                    Detection(
                        label=self.label_for(int(row[5])),
                        confidence=min(max(score, 0.0), 1.0),
                        bbox=[float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                    )
                )
            return detections

        # (C,): class logits — treat as whole-image classification.
        if raw.ndim == 1 and raw.size > 0:
            probs = _softmax(raw)
            index = int(np.argmax(probs))
            score = float(probs[index])
            if score < threshold:
                return []
            width = float(self.config.input_width)
            height = float(self.config.input_height)
            return [
                Detection(
                    label=self.label_for(index),
                    confidence=min(max(score, 0.0), 1.0),
                    bbox=[0.0, 0.0, width, height],
                )
            ]

        logger.warning(
            "Unrecognised model output shape {} — implement postprocess() for your model",
            raw.shape,
        )
        return []

    def label_for(self, index: int) -> str:
        """Map a class index to a human-readable label."""
        if 0 <= index < len(self.config.labels):
            return self.config.labels[index]
        return f"class_{index}"

    # ----------------------------------------------------------------- inference

    def predict(
        self,
        image: Image.Image,
        confidence_threshold: float | None = None,
    ) -> tuple[list[Detection], float]:
        """Run the full pipeline. Returns ``(detections, inference_time_ms)``.

        Raises :class:`RuntimeError` if no model is loaded — callers should check
        :attr:`is_ready` first and answer 503.
        """
        session = self._session
        if session is None or self._input_name is None:
            raise RuntimeError("no ONNX model is loaded")

        threshold = (
            self.config.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        )
        batch = self.preprocess(image)

        started = time.perf_counter()
        outputs: list[Any] = session.run(None, {self._input_name: batch})
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        detections = self.postprocess(outputs, threshold)
        logger.debug(
            "Inference on '{}' produced {} detection(s) in {:.2f} ms",
            self.config.model_name,
            len(detections),
            elapsed_ms,
        )
        return detections, elapsed_ms


def _softmax(values: FloatArray) -> FloatArray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    result: FloatArray = (exponentials / np.sum(exponentials)).astype(np.float32)
    return result

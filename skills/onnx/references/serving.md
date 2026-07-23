# Serving ONNX Models

Scope: exposing an ONNX Runtime session over HTTP with FastAPI, loading the session once
at startup and reusing it across requests.

## Serving with ONNX Runtime and FastAPI

Load the session once in a `lifespan` handler; keep it in module scope for reuse across
requests.

```python
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]
    inference_time_ms: float = Field(ge=0)

# Global session
session: ort.InferenceSession | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model on startup."""
    global session
    session = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    yield
    session = None

app = FastAPI(title="ONNX Detection API", lifespan=lifespan)

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    """Run object detection on an uploaded image."""
    import time
    import cv2

    # Read and preprocess image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    input_tensor = preprocess(image)  # Your preprocessing function

    # Run inference
    start = time.perf_counter()
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    inference_time = (time.perf_counter() - start) * 1000

    # Parse outputs
    boxes, scores, labels = postprocess(outputs)  # Your postprocessing function

    return PredictionResponse(
        boxes=boxes.tolist(),
        scores=scores.tolist(),
        labels=labels.tolist(),
        inference_time_ms=inference_time,
    )
```

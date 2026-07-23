# gr.Interface and gr.Blocks

Scope: the two top-level demo constructors — a single-function `gr.Interface`, and a multi-tab `gr.Blocks` layout with upload, webcam streaming, and batch processing.

## Contents

- [gr.Interface — single-function demos](#grinterface--single-function-demos)
- [gr.Blocks — complex layouts](#grblocks--complex-layouts)

## gr.Interface — single-function demos

Fastest way to wrap one prediction function with defined inputs/outputs.

```python
"""Image classification demo with gr.Interface."""

from __future__ import annotations

import gradio as gr
import numpy as np
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class ClassificationConfig(BaseModel, frozen=True):
    model_path: str = "models/classifier.onnx"
    labels_path: str = "models/labels.txt"
    top_k: int = Field(default=5, ge=1, le=20)
    image_size: tuple[int, int] = (224, 224)


def classify_image(
    image: Image.Image,
    config: ClassificationConfig = ClassificationConfig(),
) -> dict[str, float]:
    """Return top-k predictions for a single image."""
    logger.info("Received image of size {}", image.size)
    preprocessed = preprocess(image, config.image_size)
    predictions = run_model(preprocessed, config.model_path)
    labels = load_labels(config.labels_path)

    top_indices = np.argsort(predictions)[-config.top_k :][::-1]
    results = {labels[i]: float(predictions[i]) for i in top_indices}
    logger.info("Top prediction: {} ({:.3f})", *next(iter(results.items())))
    return results


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="Image Classifier",
    description="Upload an image to classify it.",
    examples=[["examples/cat.jpg"], ["examples/dog.jpg"]],
    cache_examples=True,
)
```

## gr.Blocks — complex layouts

Standard for production demos. Combines tabs, rows/columns, buttons, streaming,
and batch inputs. The pattern below shows upload, webcam streaming, and batch tabs.

```python
"""Object detection demo with gr.Blocks layout."""

from __future__ import annotations

import gradio as gr
from loguru import logger
from pydantic import BaseModel, Field


class DetectionConfig(BaseModel, frozen=True):
    model_path: str = "models/detector.onnx"
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1, le=500)
    device: str = "cpu"


def build_detection_demo(config: DetectionConfig | None = None) -> gr.Blocks:
    config = config or DetectionConfig()
    logger.info("Building detection demo with model: {}", config.model_path)

    with gr.Blocks(title="Object Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Object Detection Demo")

        with gr.Tab("Image Upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Input Image")
                    confidence_slider = gr.Slider(
                        0.0, 1.0, value=config.confidence_threshold,
                        step=0.05, label="Confidence Threshold",
                    )
                    detect_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column(scale=1):
                    output_image = gr.Image(type="pil", label="Detections")
                    output_json = gr.JSON(label="Detection Results")
            detect_btn.click(
                fn=run_detection,
                inputs=[input_image, confidence_slider],
                outputs=[output_image, output_json],
            )

        with gr.Tab("Webcam"):  # .stream() for real-time per-frame processing
            webcam_input = gr.Image(sources=["webcam"], type="pil", label="Webcam Feed")
            webcam_output = gr.Image(type="pil", label="Live Detections")
            webcam_input.stream(
                fn=run_detection_realtime, inputs=[webcam_input], outputs=[webcam_output],
            )

        with gr.Tab("Batch Processing"):  # gr.File(multiple) + gr.Gallery output
            batch_input = gr.File(file_count="multiple", file_types=["image"], label="Images")
            batch_output = gr.Gallery(label="Results", columns=3)
            batch_btn = gr.Button("Process Batch", variant="primary")
            batch_btn.click(
                fn=run_batch_detection,
                inputs=[batch_input, confidence_slider],
                outputs=[batch_output],
            )

    return demo
```

# Model Serving in Gradio

Scope: loading a Hub model with `gr.load`, and wrapping custom ONNX or PyTorch checkpoints in a server class loaded once at startup.

## Contents

- [From the Hugging Face Hub with gr.load](#from-the-hugging-face-hub-with-grload)
- [Custom ONNX / PyTorch model server](#custom-onnx--pytorch-model-server)
- [PyTorch model with @torch.inference_mode](#pytorch-model-with-torchinference_mode)

## From the Hugging Face Hub with gr.load

```python
demo = gr.load(
    name="facebook/detr-resnet-50",
    src="models",
    title="DETR Object Detection",
    description="Detect objects using DETR.",
)
```

## Custom ONNX / PyTorch model server

Wrap the model in a class loaded once at startup; reference it via closure in the
callback. Same pattern applies to a PyTorch checkpoint (swap `ort.InferenceSession`
for `torch.load(...).eval()` and preprocess with `torchvision.transforms`).

```python
"""Serve a custom ONNX model through Gradio."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field


class ONNXModelConfig(BaseModel, frozen=True):
    model_path: Path = Path("models/model.onnx")
    input_size: tuple[int, int] = (640, 640)
    providers: list[str] = Field(
        default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )


class ModelServer:
    """Wraps an ONNX model for Gradio serving (loaded once at startup)."""

    def __init__(self, config: ONNXModelConfig) -> None:
        self.config = config
        logger.info("Loading ONNX model from {}", config.model_path)
        self.session = ort.InferenceSession(str(config.model_path), providers=config.providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """Run inference, return annotated image and results dict."""
        input_array = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_array})
        detections = self._postprocess(outputs, image.size)
        annotated = self._draw_detections(image, detections)
        logger.info("Found {} detections", len(detections))
        return annotated, {"num_detections": len(detections), "detections": detections}

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize(self.config.input_size)
        array = np.array(resized, dtype=np.float32) / 255.0
        return np.transpose(array, (2, 0, 1))[np.newaxis, ...]

    def _postprocess(self, outputs: list, original_size: tuple[int, int]) -> list[dict]:
        ...  # depends on model architecture

    def _draw_detections(self, image: Image.Image, detections: list[dict]) -> Image.Image:
        ...  # uses PIL.ImageDraw


def create_onnx_demo(config: ONNXModelConfig | None = None) -> gr.Blocks:
    server = ModelServer(config or ONNXModelConfig())
    with gr.Blocks(title="ONNX Model Demo") as demo:
        gr.Markdown("# Custom ONNX Model Demo")
        with gr.Row():
            input_img = gr.Image(type="pil", label="Input")
            output_img = gr.Image(type="pil", label="Output")
        results_json = gr.JSON(label="Results")
        gr.Button("Run Inference", variant="primary").click(
            fn=server.predict, inputs=[input_img], outputs=[output_img, results_json],
        )
    return demo
```

## PyTorch model with @torch.inference_mode

```python
from torchvision import transforms


def create_pytorch_demo(config: TorchModelConfig) -> gr.Blocks:
    model = torch.load(config.checkpoint_path, map_location=config.device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(config.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @torch.inference_mode()
    def predict(image: Image.Image) -> dict[str, float]:
        tensor = transform(image).unsqueeze(0).to(config.device)
        probs = torch.nn.functional.softmax(model(tensor)[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)
        return {
            (config.class_names[idx] if config.class_names else f"class_{idx}"): float(prob)
            for prob, idx in zip(top5_prob, top5_idx)
        }

    with gr.Blocks(title="PyTorch Model Demo") as demo:
        with gr.Row():
            input_img = gr.Image(type="pil", label="Upload Image")
            output_label = gr.Label(num_top_classes=5, label="Predictions")
        gr.Button("Classify", variant="primary").click(
            fn=predict, inputs=[input_img], outputs=[output_label],
        )
    return demo
```

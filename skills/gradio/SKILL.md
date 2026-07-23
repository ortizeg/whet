---
name: gradio
description: >
  Use this skill when building an interactive demo or lightweight UI to try an ML model —
  gr.Interface and gr.Blocks layouts, image/video/text inputs and outputs, loading models
  with gr.load, custom components, flagging and feedback collection, and deploying to
  Hugging Face Spaces. Reach for it any time you'd otherwise hand-build a front-end so
  people can play with a model, even if the user just says "make a UI to test the model"
  or "put up a demo". For a production JSON serving API instead of a demo, see fastapi.
---

# Gradio Skill

Build Gradio 4.x demos with typed interfaces, Pydantic-validated configs, and
Loguru logging. Use `gr.Interface` for a single prediction function; use
`gr.Blocks` for multi-step workflows, comparisons, tabs, and conditional
visibility. Never expose raw model internals to the UI layer.

## gr.Interface — Single-Function Demos

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

## gr.Blocks — Complex Layouts

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

## Multi-Modal Inputs and Outputs

Mix `gr.Image`, `gr.Video`, `gr.Textbox`, `gr.Gallery`, and `gr.JSON`. Wire events
with `.change()` (fire on input change) or `.click()` (fire on button).

```python
def build_multimodal_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Modal Analysis")

        with gr.Tab("Image Captioning"):
            img_input = gr.Image(type="pil", label="Upload Image")
            caption_output = gr.Textbox(label="Generated Caption", lines=3)
            img_input.change(fn=generate_caption, inputs=[img_input], outputs=[caption_output])

        with gr.Tab("Video Analysis"):  # one input -> multiple outputs
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Video(label="Annotated Video")
            frame_gallery = gr.Gallery(label="Key Frames", columns=4)
            analysis_json = gr.JSON(label="Analysis Results")
            gr.Button("Analyze Video", variant="primary").click(
                fn=analyze_video,
                inputs=[video_input],
                outputs=[video_output, frame_gallery, analysis_json],
            )

        with gr.Tab("Visual Question Answering"):  # multiple inputs -> one output
            with gr.Row():
                vqa_image = gr.Image(type="pil", label="Image")
                with gr.Column():
                    vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?")
                    vqa_answer = gr.Textbox(label="Answer", interactive=False)
                    gr.Button("Ask", variant="primary").click(
                        fn=answer_question, inputs=[vqa_image, vqa_question], outputs=[vqa_answer],
                    )

    return demo
```

## Model Serving

### From the Hugging Face Hub with gr.load

```python
demo = gr.load(
    name="facebook/detr-resnet-50",
    src="models",
    title="DETR Object Detection",
    description="Detect objects using DETR.",
)
```

### Custom ONNX / PyTorch Model Server

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

### PyTorch Model with @torch.inference_mode

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

## Reusable Components and Comparison Layouts

Factor repeated controls into helper functions returning components. Reuse them to
build side-by-side comparison layouts.

```python
def create_model_selector(models: dict[str, str], default: str | None = None) -> gr.Dropdown:
    choices = list(models.keys())
    return gr.Dropdown(choices=choices, value=default or choices[0], label="Select Model")


def create_preprocessing_controls() -> tuple[gr.Slider, gr.Slider, gr.Checkbox]:
    brightness = gr.Slider(-1.0, 1.0, value=0.0, step=0.1, label="Brightness")
    contrast = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Contrast")
    grayscale = gr.Checkbox(value=False, label="Convert to Grayscale")
    return brightness, contrast, grayscale


def build_comparison_layout() -> gr.Blocks:
    with gr.Blocks() as demo:
        input_image = gr.Image(type="pil", label="Input Image")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A")
                model_a = create_model_selector({"YOLOv8-n": "yolov8n", "YOLOv8-s": "yolov8s"})
                output_a = gr.Image(type="pil", label="Model A Output")
                metrics_a = gr.JSON(label="Model A Metrics")
            with gr.Column():
                gr.Markdown("### Model B")
                model_b = create_model_selector({"YOLOv8-m": "yolov8m", "YOLOv8-l": "yolov8l"})
                output_b = gr.Image(type="pil", label="Model B Output")
                metrics_b = gr.JSON(label="Model B Metrics")
        gr.Button("Compare Models", variant="primary").click(
            fn=compare_models,
            inputs=[input_image, model_a, model_b],
            outputs=[output_a, metrics_a, output_b, metrics_b],
        )
    return demo
```

## Flagging and Feedback Collection

Subclass `gr.FlaggingCallback` to persist structured feedback for active learning.

```python
"""Custom flagging callback for ML feedback collection."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from loguru import logger
from pydantic import BaseModel


class FlaggedSample(BaseModel, frozen=True):
    timestamp: str
    flag_reason: str
    input_hash: str
    prediction: str
    user_correction: str | None = None


class MLFlaggingCallback(gr.FlaggingCallback):
    def __init__(self, output_dir: str = "flagged_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, components: list, flagging_dir: str | Path) -> None:
        self.flagging_dir = Path(flagging_dir)
        self.flagging_dir.mkdir(parents=True, exist_ok=True)

    def flag(self, flag_data: list, flag_option: str = "incorrect", username: str | None = None) -> int:
        sample = FlaggedSample(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            flag_reason=flag_option,
            input_hash=str(hash(str(flag_data[0]))),
            prediction=str(flag_data[1]) if len(flag_data) > 1 else "",
            user_correction=str(flag_data[2]) if len(flag_data) > 2 else None,
        )
        (self.output_dir / f"flag_{sample.timestamp}.json").write_text(sample.model_dump_json(indent=2))
        logger.info("Flagged sample saved (reason: {})", flag_option)
        return 1


demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    flagging_callback=MLFlaggingCallback(output_dir="flagged_data"),
    flagging_options=["incorrect", "low_confidence", "interesting"],
)
```

## Deployment

Launch with production settings; optionally add auth or mount inside FastAPI.
Never use `share=True` in production — host on Spaces, Docker, or behind a proxy.

```python
def launch_demo(demo: gr.Blocks, share: bool = False) -> None:
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True,
        max_threads=10,
        auth=None,  # or [("admin", "secure_password")] for basic auth
    )


def mount_on_fastapi(demo: gr.Blocks):
    from fastapi import FastAPI

    app = FastAPI(title="ML Demo API")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return gr.mount_gradio_app(app, demo, path="/demo")
```

### Hugging Face Spaces

`Dockerfile` and `app.py` entry point:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:${PATH}"
COPY pixi.toml pixi.lock ./
RUN pixi install
COPY src/ ./src/
COPY models/ ./models/
EXPOSE 7860
CMD ["pixi", "run", "python", "-m", "src.app", "--server-name", "0.0.0.0", "--server-port", "7860"]
```

```python
"""app.py — Hugging Face Spaces entry point."""

from src.config import DetectionConfig
from src.demo import build_detection_demo

demo = build_detection_demo(DetectionConfig())

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

## Anti-Patterns

- **Never load models inside callbacks** -- load once at startup and reference via closure or a server class. Reloading per request adds massive latency.
- **Never use `gr.Interface` for multi-step workflows** -- use `gr.Blocks` for explicit layout control.
- **Never expose raw tensors/numpy to outputs** -- convert to PIL Images, JSON-serializable dicts, or strings first.
- **Never hardcode file paths** -- use Pydantic config models so paths are environment-configurable.
- **Never skip input validation** -- check image sizes, file types, and parameter ranges before inference.
- **Never use `share=True` in production** -- use proper hosting.
- **Never block the main thread** on long inference -- use `gr.Progress` and consider async patterns.
- **Never skip logging** -- use Loguru in prediction functions, model loading, and error handling.

## Integration with Other Skills

Pydantic (frozen config models), Loguru (structured logging), Hugging Face
(load Hub models, deploy to Spaces), FastAPI (mount demos), ONNX (low-latency
serving), PyTorch Lightning (load checkpoints), Testing (Gradio test client + pytest).

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
visibility. Never expose raw model internals to the UI layer. This page carries
the two core constructors; the deep dives hold serving, feedback, and deployment.

## Core: `gr.Interface` for one prediction function

Wrap a single typed function with declared inputs and outputs. This covers most
"let me try the model" requests.

```python
from __future__ import annotations

import gradio as gr
from loguru import logger
from PIL import Image


def classify_image(image: Image.Image) -> dict[str, float]:
    """Return label -> probability for a single image."""
    logger.info("Received image of size {}", image.size)
    return run_model(image)  # JSON-serializable dict, never raw tensors


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

## Core: `gr.Blocks` when the demo has more than one step

`gr.Blocks` gives explicit layout control — tabs, rows, columns, and events wired
per component. Load the model once outside the callback and close over it.

```python
def build_demo(config: DetectionConfig) -> gr.Blocks:
    server = ModelServer(config)  # loaded once at startup

    with gr.Blocks(title="Object Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Object Detection Demo")
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Input Image")
                confidence = gr.Slider(
                    0.0, 1.0, value=config.confidence_threshold,
                    step=0.05, label="Confidence Threshold",
                )
                detect_btn = gr.Button("Detect Objects", variant="primary")
            with gr.Column(scale=1):
                output_image = gr.Image(type="pil", label="Detections")
                output_json = gr.JSON(label="Detection Results")

        detect_btn.click(
            fn=server.predict,
            inputs=[input_image, confidence],
            outputs=[output_image, output_json],
        )

    return demo


demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
```

## Conventions

- **Load models once at startup** — instantiate a server class or module-level object and reference it from callbacks via closure.
- **Configure with frozen Pydantic models** — model paths, thresholds, and input sizes belong in a `BaseModel(frozen=True)`, not in literals scattered through the layout.
- **Type every callback** — annotated inputs and returns make the component wiring checkable and keep the UI layer honest.
- **Return UI-native types** — PIL Images, JSON-serializable dicts, strings; convert tensors and numpy arrays before returning.
- **Log with Loguru** in prediction functions, model loading, and error handling.
- **Build layouts inside the `with gr.Blocks()` block** — components register with the enclosing context at construction time.
- **Keep `outputs=[...]` ordered to match the return tuple** — Gradio maps positionally, not by name.
- **Validate inputs before inference** — image sizes, file types, and parameter ranges.
- **Bind `0.0.0.0:7860`** so local runs and Hugging Face Spaces behave identically.

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

## Deep dives

- `references/interface-and-blocks.md` — read when you need the full `gr.Interface` example with config, or a multi-tab `gr.Blocks` demo with webcam streaming and batch processing.
- `references/inputs-and-outputs.md` — read when mixing image/video/text components, choosing `.change()` vs `.click()`, or fanning one input into several outputs.
- `references/model-serving.md` — read when serving a Hub model with `gr.load`, or wrapping a custom ONNX session or PyTorch checkpoint in a startup-loaded server class.
- `references/reusable-components.md` — read when factoring repeated controls into helpers or building side-by-side model comparison layouts.
- `references/flagging-and-feedback.md` — read when collecting user corrections for active learning via a custom `gr.FlaggingCallback`.
- `references/deployment-and-spaces.md` — read when configuring `launch()` for production, mounting the demo inside FastAPI, or shipping to Hugging Face Spaces with Docker.

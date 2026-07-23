# Reusable Components and Comparison Layouts

Scope: factoring repeated controls into helper functions that return components, and using them to build side-by-side model comparison layouts.

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

Helper functions must be called *inside* the `gr.Blocks` context — Gradio components
register themselves with the enclosing block at construction time, so a component
built outside the `with` block will not appear in the demo.

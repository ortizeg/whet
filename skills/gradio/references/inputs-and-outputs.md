# Multi-Modal Inputs and Outputs

Scope: mixing image/video/text components, wiring `.change()` vs `.click()` events, and fanning one input out to several outputs (or several inputs into one).

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

Key points:

- The order of `outputs=[...]` must match the order of the tuple your function returns.
- `.change()` fires on every edit, so avoid it for expensive inference — use a button `.click()` instead.
- Mark display-only textboxes `interactive=False` so users cannot type into results.
- Always convert model output to PIL Images, JSON-serializable dicts, or strings before returning — never raw tensors or numpy arrays.

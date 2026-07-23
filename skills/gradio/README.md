# Gradio Skill

The Gradio Skill provides expert patterns for building interactive ML demos and model serving UIs with Gradio 4.x. It covers `gr.Interface` for simple demos, `gr.Blocks` for complex multi-tab layouts, multi-modal inputs and outputs (image, video, text), ONNX and PyTorch model integration, custom flagging for feedback collection, and deployment to Hugging Face Spaces and Docker.

## Purpose

When you need to quickly build an interactive demo for a trained model -- whether for stakeholder presentations, user testing, dataset labeling feedback, or public deployment -- Gradio is the standard tool. This skill encodes best practices for structuring Gradio applications in ML/CV contexts: proper model lifecycle management at startup, typed Pydantic configurations, reusable component patterns, and production deployment strategies including Hugging Face Spaces and FastAPI mounting.

## When to Use

- When building interactive demos for image classification, object detection, or segmentation models.
- When you need a quick web UI for model prototyping and stakeholder presentations.
- When collecting user feedback on model predictions via flagging for active learning.
- When deploying public-facing demos to Hugging Face Spaces.
- When creating side-by-side model comparison tools for evaluation.
- When serving real-time webcam inference through a browser-based interface.
- When mounting an ML demo inside an existing FastAPI application.

## Key Features

- **gr.Interface** -- single-function demos with automatic input/output type inference and example caching.
- **gr.Blocks** -- complex layouts with tabs, rows, columns, and conditional visibility for multi-step workflows.
- **Multi-modal I/O** -- image, video, text, audio, JSON, and gallery components for diverse ML tasks.
- **Model serving** -- ONNX and PyTorch model loading at startup with Pydantic-configured inference pipelines.
- **gr.load** -- instant demos from Hugging Face Hub models without writing inference code.
- **Custom flagging** -- structured feedback collection for active learning and dataset improvement.
- **Deployment** -- Hugging Face Spaces, Docker, share links, authentication, and FastAPI mounting.
- **Reusable components** -- model selectors, preprocessing controls, and comparison layouts.

## Related Skills

- **[FastAPI](../fastapi/)** -- mount Gradio demos inside FastAPI apps for combined API + demo serving.
- **[Hugging Face](../huggingface/)** -- load pretrained models from the Hub and deploy to Hugging Face Spaces.
- **[PyTorch Lightning](../pytorch-lightning/)** -- load Lightning checkpoints and serve trained models through Gradio.
- **[Testing](../testing/)** -- test Gradio components with the Gradio test client and pytest fixtures.
- **[ONNX](../onnx/)** -- serve optimized ONNX models for low-latency Gradio inference.
- **[Pydantic](../pydantic/)** -- frozen BaseModel patterns for all Gradio configuration objects.
- **[Loguru](../loguru/)** -- structured logging in model loading, prediction callbacks, and flagging.

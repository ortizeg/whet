# Deployment and Hugging Face Spaces

Scope: launch settings for production, mounting a demo inside FastAPI, and packaging the demo for Hugging Face Spaces with Docker.

## Contents

- [Launching](#launching)
- [Mounting inside FastAPI](#mounting-inside-fastapi)
- [Hugging Face Spaces](#hugging-face-spaces)

## Launching

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
```

## Mounting inside FastAPI

```python
def mount_on_fastapi(demo: gr.Blocks):
    from fastapi import FastAPI

    app = FastAPI(title="ML Demo API")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    return gr.mount_gradio_app(app, demo, path="/demo")
```

This gives one process serving both a JSON API and the demo UI — useful when the
same model must back a production endpoint and a human-facing playground.

## Hugging Face Spaces

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

Spaces expects port 7860 and binds `0.0.0.0`; keeping the same values in
`launch_demo` means the local run and the deployed Space behave identically.

# CV Inference Service

Production deployment of trained models using FastAPI and ONNX Runtime, with Docker containerization and health monitoring.

## Purpose

This archetype packages trained models into production-ready REST API services. It uses ONNX Runtime for optimized inference, FastAPI for the API layer, Pydantic for request/response validation, and Docker for deployment.

## Directory Structure

```
${project_slug}/
├── models/
│   └── .gitkeep
├── src/
│   └── ${package_name}/
│       ├── inference/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── __init__.py
│       ├── app.py
│       ├── config.py
│       ├── py.typed
│       └── schemas.py
├── tests/
│   ├── __init__.py
│   └── test_app.py
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── pixi.toml
└── pyproject.toml
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Run inference on image |
| POST | `/predict/batch` | Batch inference |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |

## Usage

```bash
# Run locally
uvicorn my_project.serve:app --reload

# Build and run with Docker
docker compose up inference

# Test the API
curl -X POST http://localhost:8000/predict \
  -F "image=@test.jpg"
```

## Customization

- Add preprocessing pipelines in `inference/preprocessing.py`
- Define new endpoints in `serve.py`
- Add model variants by extending the predictor abstraction
- Configure GPU/CPU inference via environment variables

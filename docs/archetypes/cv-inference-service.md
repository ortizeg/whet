# CV Inference Service

Production deployment of trained models using FastAPI and ONNX Runtime, with Docker containerization and health monitoring.

## Purpose

This archetype packages trained models into production-ready REST API services. It uses ONNX Runtime for optimized inference, FastAPI for the API layer, Pydantic for request/response validation, and Docker for deployment.

## Directory Structure

```
{{project_slug}}/
├── src/{{package_name}}/
│   ├── __init__.py
│   ├── serve.py               # FastAPI application
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py       # ONNX Runtime wrapper
│   │   └── preprocessing.py   # Input transforms
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py         # Pydantic request models
│   │   └── response.py        # Pydantic response models
│   └── utils/
│       └── health.py          # Health check endpoints
├── models/                    # ONNX model files
│   └── .gitkeep
├── Dockerfile                 # Multi-stage (slim inference)
├── docker-compose.yml         # GPU support + monitoring
├── tests/
│   ├── test_api.py
│   ├── test_predictor.py
│   └── test_preprocessing.py
└── ...
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

# Serving Models

Scope: exposing a logged or registered MLflow model as a REST endpoint and calling it.

MLflow can serve models as REST APIs:

```bash
# Serve a registered model
mlflow models serve \
    --model-uri "models:/yolov8-coco/Production" \
    --port 8000 \
    --no-conda

# Make predictions
curl -X POST http://localhost:8000/invocations \
    -H "Content-Type: application/json" \
    -d '{"inputs": [...]}'
```

Notes:

- The `--model-uri` accepts the same URI forms as `load_model`: `models:/name/Stage`, `models:/name/version`, or `runs:/<run-id>/model`.
- `--no-conda` runs in the current environment; omit it to let MLflow rebuild the model's recorded environment, which is slower but reproducible.
- Built-in serving is a strength MLflow has over W&B — no separate serving stack is required for simple deployments.

# Real-Time Endpoints and Inference Handlers

Deploying a trained model as a persistent HTTPS endpoint, and the four handler functions SageMaker calls to serve requests.

## Contents

- [Deploying an Endpoint](#deploying-an-endpoint)
- [Custom Inference Handlers](#custom-inference-handlers)
- [Handler Rules](#handler-rules)

## Deploying an Endpoint

Deploy a `PyTorchModel` with a custom inference script:

```python
from pydantic import BaseModel
from sagemaker.pytorch import PyTorchModel


class EndpointConfig(BaseModel, frozen=True):
    role: str
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    endpoint_name: str
    model_data_s3: str
    framework_version: str = "2.1.0"
    py_version: str = "py310"


def deploy_endpoint(config: EndpointConfig) -> str:
    model = PyTorchModel(
        model_data=config.model_data_s3,
        role=config.role,
        framework_version=config.framework_version,
        py_version=config.py_version,
        entry_point="inference.py",
        source_dir="src/inference",
    )
    predictor = model.deploy(
        initial_instance_count=config.instance_count,
        instance_type=config.instance_type,
        endpoint_name=config.endpoint_name,
    )
    return predictor.endpoint_name
```

Notes:

- `model_data` points at the `model.tar.gz` a training job wrote to `output_path`.
- `source_dir` is uploaded and unpacked in the container; a `requirements.txt` there is
  pip-installed at container start.
- `framework_version` / `py_version` select the managed DLC image — keep them identical to the
  training job so the serialized weights load cleanly.
- Endpoints bill continuously while they exist. Delete them (`predictor.delete_endpoint()`)
  when idle, or use batch transform for offline workloads.
- Passing an existing `endpoint_name` to `deploy` creates a new endpoint; to update in place,
  create a new endpoint config and call `update_endpoint`.

## Custom Inference Handlers

SageMaker calls these in order: `model_fn` (once at load), then `input_fn → predict_fn → output_fn` per request.

```python
"""src/inference/inference.py"""

from __future__ import annotations

import io
import json

import torch
from PIL import Image
from torchvision import transforms


def model_fn(model_dir: str) -> torch.nn.Module:
    model = build_model("resnet50", num_classes=10)
    model.load_state_dict(torch.load(f"{model_dir}/model.pth", map_location="cpu"))
    model.eval()
    return model.cuda() if torch.cuda.is_available() else model


def input_fn(request_body: bytes, content_type: str) -> torch.Tensor:
    if content_type == "application/x-image":
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return tfm(image).unsqueeze(0)
    if content_type == "application/json":
        return torch.tensor(json.loads(request_body)["instances"])
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: torch.Tensor, model: torch.nn.Module) -> dict:
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(input_data.to(device))
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    return {
        "predictions": preds.cpu().numpy().tolist(),
        "probabilities": probs.cpu().numpy().tolist(),
    }


def output_fn(prediction: dict, accept: str) -> str:
    if accept == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported accept type: {accept}")
```

## Handler Rules

- **`model_fn` runs once per container**, not per request. All expensive setup — weight loading,
  compilation, warmup — belongs here. `map_location="cpu"` then `.cuda()` avoids device-mismatch
  errors when the checkpoint was saved from a GPU.
- **`model.eval()` is not optional.** Forgetting it leaves dropout and batch-norm in training
  mode and silently degrades predictions.
- **`input_fn` must dispatch on `content_type`** and raise for anything unsupported — an
  unhandled content type otherwise produces a confusing 500 rather than a clear 415-style error.
- **The preprocessing in `input_fn` must match training exactly** (same resize, same
  normalization constants). This is the single most common source of "the endpoint is worse
  than my local eval".
- **`predict_fn` wraps inference in `torch.no_grad()`** and reads the device off the model
  rather than assuming CUDA, so the same script runs on CPU instance types.
- **Return JSON-serializable Python**, never tensors or numpy arrays — hence `.cpu().numpy().tolist()`.
- The same `inference.py` is reused verbatim by batch transform, so keep it free of
  endpoint-specific assumptions.

# ONNX

The ONNX skill covers model export, optimization, and inference using the ONNX ecosystem, including ONNX Runtime and TensorRT integration for production deployment.

**Skill directory:** `skills/onnx/`

## Purpose

Training and inference have different requirements. Training uses PyTorch for flexibility; inference needs speed and portability. ONNX bridges this gap by providing a standard model format that runs on any platform with optimized runtimes. This skill teaches Claude Code to export PyTorch models to ONNX, optimize graphs, run inference with ONNX Runtime, and integrate with TensorRT for GPU-accelerated deployment.

## When to Use

- Deploying trained models to production inference services
- Cross-platform model serving (CPU, GPU, edge devices)
- Model optimization for latency-sensitive applications
- Exporting models for non-Python inference environments (C++, Rust, mobile)

## Key Patterns

### Exporting a PyTorch Model

```python
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
) -> None:
    """Export a PyTorch model to ONNX format."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
```

### ONNX Runtime Inference

```python
import numpy as np
import onnxruntime as ort

class ONNXPredictor:
    def __init__(self, model_path: str | Path) -> None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed image."""
        outputs = self.session.run(None, {self.input_name: image})
        return outputs[0]
```

### Model Validation

```python
import onnx

def validate_onnx_model(model_path: Path) -> None:
    """Validate that an ONNX model is well-formed."""
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    # Compare outputs between PyTorch and ONNX
    pytorch_output = pytorch_model(dummy_input).detach().numpy()
    onnx_output = onnx_session.run(None, {"input": dummy_input.numpy()})[0]
    np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-3, atol=1e-5)
```

### OnnxSlim Post-Export (Required)

```python
import onnx
import onnxslim

raw_model = onnx.load("model_raw.onnx")
slimmed = onnxslim.slim(raw_model)
onnx.save(slimmed, "model.onnx")
```

## Anti-Patterns to Avoid

- Do not deploy raw exported models -- always run `onnxslim.slim()` after export
- Do not export models in training mode -- always call `model.eval()` first
- Do not use fixed batch dimensions unless deployment requires it -- use dynamic axes
- Avoid exporting with old opset versions -- use opset 17+ for modern operator support
- Do not skip output validation -- always compare PyTorch and ONNX outputs numerically
- Do not quantize before slimming -- slim first for better quantization results

## Combines Well With

- **PyTorch Lightning** -- Export from trained LightningModule checkpoints
- **Docker CV** -- ONNX Runtime inference containers
- **Pydantic** -- Validated request/response schemas around inference
- **Testing** -- Numerical equivalence tests between PyTorch and ONNX

## Full Reference

See [`skills/onnx/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/onnx/SKILL.md) for patterns including TensorRT optimization, quantization, model surgery, and batched inference with dynamic shapes.

# ONNX Export and Dynamic Axes

Scope: exporting PyTorch models to ONNX with validated configuration, dynamic axis
recipes, inspecting model input/output specs, and multi-input/multi-output exports.

## Contents

- [Export with Pydantic Configuration](#export-with-pydantic-configuration)
- [Dynamic Axes Configuration](#dynamic-axes-configuration)
- [Inspecting Model Inputs and Outputs](#inspecting-model-inputs-and-outputs)
- [Multiple Inputs and Outputs](#multiple-inputs-and-outputs)

## Export with Pydantic Configuration

Always call `model.eval()` before export, run `onnx.checker.check_model()` to validate,
and use `opset_version=17`+ for modern architectures.

```python
import torch
import onnx
from pydantic import BaseModel, Field

class ONNXExportConfig(BaseModel):
    """ONNX export configuration."""
    opset_version: int = Field(ge=11, default=17)
    dynamic_axes: dict[str, dict[int, str]] | None = None
    input_names: list[str] = Field(default_factory=lambda: ["input"])
    output_names: list[str] = Field(default_factory=lambda: ["output"])

def export_to_onnx(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    output_path: str,
    config: ONNXExportConfig,
) -> None:
    """Export PyTorch model to ONNX with validated configuration."""
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=config.opset_version,
        input_names=config.input_names,
        output_names=config.output_names,
        dynamic_axes=config.dynamic_axes,
    )
    onnx.checker.check_model(onnx.load(output_path))

config = ONNXExportConfig(
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size"},
    },
)
export_to_onnx(model, torch.randn(1, 3, 640, 640), "model.onnx", config)
```

## Dynamic Axes Configuration

Dynamic axes allow the model to accept variable-size inputs at runtime. This is essential
for production deployment where batch sizes, image heights, and widths may vary.

### Common Configurations

```python
# Image classification: variable batch size
dynamic_axes = {
    "input": {0: "batch_size"},
    "output": {0: "batch_size"},
}

# Object detection: variable batch size and image size
dynamic_axes = {
    "input": {0: "batch_size", 2: "height", 3: "width"},
    "output": {0: "batch_size"},
}

# Sequence models: variable batch and sequence length
dynamic_axes = {
    "input": {0: "batch_size", 1: "sequence_length"},
    "output": {0: "batch_size", 1: "sequence_length"},
}
```

## Inspecting Model Inputs and Outputs

```python
import onnx

def inspect_model(model_path: str) -> None:
    """Print input and output specifications of an ONNX model."""
    model = onnx.load(model_path)

    print("Inputs:")
    for inp in model.graph.input:
        shape = [dim.dim_value or dim.dim_param for dim in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        print(f"  {inp.name}: {shape} ({dtype})")

    print("\nOutputs:")
    for out in model.graph.output:
        shape = [dim.dim_value or dim.dim_param for dim in out.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type)
        print(f"  {out.name}: {shape} ({dtype})")

inspect_model("model.onnx")
```

## Multiple Inputs and Outputs

```python
import torch

class MultiInputModel(torch.nn.Module):
    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        enhanced = self.metadata_encoder(metadata)
        boxes = self.detection_head(features, enhanced)
        scores = self.score_head(features, enhanced)
        return boxes, scores

# Export with multiple inputs/outputs
dummy_image = torch.randn(1, 3, 640, 640)
dummy_metadata = torch.randn(1, 10)

torch.onnx.export(
    model,
    (dummy_image, dummy_metadata),
    "multi_input_model.onnx",
    opset_version=17,
    input_names=["image", "metadata"],
    output_names=["boxes", "scores"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "metadata": {0: "batch_size"},
        "boxes": {0: "batch_size"},
        "scores": {0: "batch_size"},
    },
)
```

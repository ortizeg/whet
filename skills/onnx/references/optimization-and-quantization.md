# ONNX Optimization and Quantization

Scope: shrinking and speeding up an exported ONNX graph — OnnxSlim, ONNX Runtime graph
optimization, INT8 dynamic/static quantization, and FP16 conversion.

## Contents

- [OnnxSlim (Required)](#onnxslim-required)
- [ORT Graph Optimization](#ort-graph-optimization)
- [Quantization](#quantization)
- [FP16 Conversion](#fp16-conversion)

## OnnxSlim (Required)

OnnxSlim is a required post-export step: it reduces operators, removes redundant nodes,
and folds constants — smaller/faster models without accuracy loss. Always slim after
export, before quantization or deployment (`pixi add onnxslim`).

### Python API

```python
import onnx
import onnxslim


def export_and_slim(
    model: torch.nn.Module,
    output_path: str,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
) -> None:
    """Export PyTorch model to ONNX and slim it."""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    raw_path = output_path.replace(".onnx", "_raw.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        raw_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size"},
        },
    )

    slimmed_model = onnxslim.slim(onnx.load(raw_path))
    onnx.save(slimmed_model, output_path)
    onnx.checker.check_model(onnx.load(output_path))
```

CLI equivalent: `onnxslim model_raw.onnx model.onnx`

Required export order: `torch.onnx.export()` → `onnxslim.slim()` →
`onnx.checker.check_model()` → (optional) quantize → (optional) benchmark. Never deploy
raw exported models or quantize before slimming.

## ORT Graph Optimization

ONNX Runtime also optimizes at session creation (complementary to OnnxSlim — use both).
Set `session_options.graph_optimization_level = ORT_ENABLE_ALL` and
`session_options.optimized_model_filepath = "model_optimized.onnx"` to persist the
optimized graph.

## Quantization

Quantization reduces model size and increases inference speed by converting weights from
FP32 to INT8:

```python
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization import CalibrationDataReader

# Dynamic quantization (no calibration data needed)
quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8,
)

# Static quantization (requires calibration data)
class ImageCalibrationReader(CalibrationDataReader):
    """Provides calibration data for static quantization."""

    def __init__(self, calibration_images: list[np.ndarray]) -> None:
        self.images = iter(calibration_images)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            image = next(self.images)
            return {"input": image}
        except StopIteration:
            return None

calibration_reader = ImageCalibrationReader(calibration_images)

quantize_static(
    model_input="model.onnx",
    model_output="model_int8_static.onnx",
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    per_channel=True,
    weight_type=QuantType.QInt8,
)
```

## FP16 Conversion

```python
import onnx
from onnxconverter_common import float16

model_fp16 = float16.convert_float_to_float16(onnx.load("model.onnx"))
onnx.save(model_fp16, "model_fp16.onnx")
```

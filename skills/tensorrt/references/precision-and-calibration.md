# Precision Modes and INT8 Calibration

Choosing FP32/FP16/INT8, and generating the calibration cache INT8 requires to preserve accuracy.

## Contents

- [Choosing a Precision](#choosing-a-precision)
- [INT8 Calibration with trtexec](#int8-calibration-with-trtexec)
- [Python Calibrator](#python-calibrator)
- [Calibration Notes](#calibration-notes)

## Choosing a Precision

- **FP32** — the baseline. Use only to establish reference outputs, or when a model is
  numerically fragile.
- **FP16** — the default for GPU inference. Roughly halves memory and typically 1.5–2x faster,
  with negligible accuracy loss on most CV models. Guard with `builder.platform_has_fast_fp16`.
- **INT8** — the fastest, but only viable **with calibration data**. Uncalibrated INT8 loses
  significant accuracy. Always validate INT8 outputs against the FP32/ONNX reference before
  shipping.

## INT8 Calibration with trtexec

INT8 gives the fastest inference but requires calibration data to preserve accuracy.

```bash
# Generate calibration cache from a dataset
trtexec --onnx=model.onnx --saveEngine=model_int8.engine \
    --int8 --calib=calibration_cache.bin --calibBatchSize=32
```

## Python Calibrator

Implement `trt.IInt8EntropyCalibrator2` to feed representative images from your own dataset.
The calibrator streams batches to TensorRT, which records activation ranges per layer and
writes them to a cache file for reuse.

```python
import numpy as np
import tensorrt as trt


class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibration using representative images."""

    def __init__(
        self,
        calibration_images: list[np.ndarray],
        batch_size: int = 8,
        cache_file: str = "calibration.cache",
    ) -> None:
        super().__init__()
        self.images = calibration_images
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0

        # Allocate device memory for one batch
        self.batch_data = np.zeros(
            (batch_size, *calibration_images[0].shape), dtype=np.float32
        )
        _, self.device_input = cudart.cudaMalloc(self.batch_data.nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.current_index >= len(self.images):
            return None

        end = min(self.current_index + self.batch_size, len(self.images))
        batch = self.images[self.current_index : end]
        self.current_index = end

        # Pad if necessary
        self.batch_data[: len(batch)] = np.stack(batch)

        cudart.cudaMemcpy(
            self.device_input,
            self.batch_data.ctypes.data,
            self.batch_data.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )

        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        from pathlib import Path

        cache = Path(self.cache_file)
        if cache.exists():
            return cache.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        from pathlib import Path

        Path(self.cache_file).write_bytes(cache)
```

## Calibration Notes

- **`get_batch` returns `None` to signal the end** of the calibration set — that is how
  TensorRT knows to stop.
- **Images must be preprocessed exactly as at inference time** (same resize, normalization,
  channel order). A mismatch produces wrong activation ranges and a badly quantized engine.
- **A few hundred representative images is usually enough.** They must cover the real input
  distribution — not just easy or synthetic examples.
- **The cache makes rebuilds cheap.** `read_calibration_cache` short-circuits the calibration
  pass on subsequent builds; delete the cache when the model or preprocessing changes.
- Attach the calibrator to the builder config with `config.int8_calibrator = ImageCalibrator(...)`
  and set `config.set_flag(trt.BuilderFlag.INT8)`.

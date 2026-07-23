# Dynamic Shapes and Optimization Profiles

Building engines that accept a range of batch sizes or input resolutions, via `trtexec` shape flags or a Python optimization profile.

## trtexec

```bash
# Variable batch size (1 to 32, optimize for 8)
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640

# Variable batch and image size
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.engine \
    --fp16 \
    --minShapes=input:1x3x320x320 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x1280x1280
```

## Python API

Same builder as in the engine-building reference, but add an optimization profile to the
config before building. `opt_shape` should match your most common batch/resolution:

```python
profile = builder.create_optimization_profile()
profile.set_shape(
    "input",
    min=(1, 3, 320, 320),
    opt=(8, 3, 640, 640),
    max=(32, 3, 1280, 1280),
)
config.add_optimization_profile(profile)
```

## Notes

- The tensor name (`"input"` above) must match the ONNX graph input name exactly. Inspect it
  with `trtexec --onnx=model.onnx --verbose` or Netron if unsure.
- **`opt_shape` is what gets tuned.** TensorRT selects kernels for the optimal shape; shapes
  near `min` or `max` still run but slower. Set `opt` to your real production batch and
  resolution, not to the midpoint of the range.
- **Keep the range tight.** A very wide min→max span forces conservative kernel choices and
  larger workspace, costing throughput across the whole range.
- The ONNX model itself must have been exported with dynamic axes for the dimensions you want
  to vary — see the ONNX skill's `dynamic_axes` argument to `torch.onnx.export`.
- At inference time with the raw Python API, call `context.set_input_shape(name, shape)` before
  execution so the runtime knows the actual dimensions for this call.

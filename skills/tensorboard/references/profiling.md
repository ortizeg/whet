# Profiling Training with the PyTorch Profiler

Scope: capturing a profiler trace during training and viewing it in TensorBoard's PyTorch Profiler tab.

TensorBoard's profiling plugin helps identify performance bottlenecks:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# Profile training
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=tensorboard_trace_handler("logs/profiler"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= (1 + 1 + 3) * 1:
            break
        with record_function("train_step"):
            loss = train_step(model, batch, optimizer)
        prof.step()

# View in TensorBoard under the "PyTorch Profiler" tab
```

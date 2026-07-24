# Model Graph, Histograms, and Custom Summaries

Scope: visualizing the computation graph, tracking weight/gradient/activation distributions, and writing text, embedding, and PR-curve summaries.

## Contents

- [Histogram Logging](#histogram-logging)
- [Graph Visualization](#graph-visualization)
- [Custom Summary Writing](#custom-summary-writing)

## Histogram Logging

Histograms show the distribution of values over time, useful for monitoring weights and gradients.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log weight and gradient distributions
for name, param in model.named_parameters():
    writer.add_histogram(f"weights/{name}", param.data, epoch)
    if param.grad is not None:
        writer.add_histogram(f"gradients/{name}", param.grad, epoch)

# Log activation distributions
def hook_fn(module, input, output, name, writer, step):
    writer.add_histogram(f"activations/{name}", output.detach(), step)

# Register hooks
for name, module in model.named_modules():
    module.register_forward_hook(
        lambda m, i, o, n=name: hook_fn(m, i, o, n, writer, global_step)
    )
```

## Graph Visualization

TensorBoard can visualize the computation graph of your model.

```python
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log model graph
dummy_input = torch.randn(1, 3, 640, 640)
writer.add_graph(model, dummy_input)
writer.close()
```

## Custom Summary Writing

For advanced use cases, write custom summaries:

```python
from torch.utils.tensorboard import SummaryWriter
import json

writer = SummaryWriter(log_dir="logs/experiment_001")

# Log text
writer.add_text("config", json.dumps(config_dict, indent=2), 0)
writer.add_text("notes", "Baseline experiment with default augmentation", 0)

# Log embeddings (useful for feature visualization)
features = model.extract_features(images)  # (N, D)
metadata = [class_names[label] for label in labels]
writer.add_embedding(
    features,
    metadata=metadata,
    label_img=images,
    global_step=epoch,
    tag="feature_embeddings",
)

# Log precision-recall curve
writer.add_pr_curve("PR/car", labels_car, predictions_car, epoch)
```

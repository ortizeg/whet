# Training Job Entry Points

The `train.py` script that runs inside a SageMaker training container: how it receives paths and hyperparameters, and how it sets up distributed training.

## Contents

- [Entry Point Script](#entry-point-script)
- [SageMaker Environment Variables](#sagemaker-environment-variables)
- [Distributed Training](#distributed-training)
- [Instance Selection](#instance-selection)

## Entry Point Script

SageMaker injects channels and paths via `SM_*` environment variables. Read them as argparse defaults, never hardcode paths.

```python
"""src/training/train.py"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--model-name", type=str, default="resnet50")
    # SageMaker-injected paths
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    p.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    logger.info("Starting training: {}", vars(args))
    world_size = int(os.environ.get("SM_NUM_GPUS", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, args.num_classes).to(device)
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # ... training loop ...

    torch.save(model.state_dict(), Path(args.model_dir) / "model.pth")
    metrics = {"final_val_loss": 0.25, "final_val_acc": 0.92}
    (Path(args.output_data_dir) / "metrics.json").write_text(json.dumps(metrics))


if __name__ == "__main__":
    train(parse_args())
```

## SageMaker Environment Variables

Hyperparameters passed to the estimator arrive as **command-line flags** with the same names,
which is why `hyperparameters=hp.model_dump()` on the estimator and `--batch-size` in argparse
line up (SageMaker converts the key `batch-size` to `--batch-size`). Use hyphens in the
`HyperParameters` model field names when they must match CLI flags.

| Variable | Meaning |
| --- | --- |
| `SM_MODEL_DIR` | Where to write the final model. Contents are tarred to `output_path` on S3. |
| `SM_CHANNEL_<NAME>` | Local path where the input channel `<name>` from `estimator.fit(inputs=...)` was downloaded. |
| `SM_OUTPUT_DATA_DIR` | Where to write non-model outputs (metrics, plots). Also uploaded to S3. |
| `SM_NUM_GPUS` | GPU count on this instance. |
| `LOCAL_RANK` | Rank of this process within the instance, set by the distributed launcher. |

Reading them as argparse **defaults** (rather than directly) means the same script runs
locally by passing flags explicitly — this is what makes local-mode testing possible.

## Distributed Training

- The estimator enables `distribution={"torch_distributed": {"enabled": True}}` only when
  `instance_count > 1`; passing it for a single instance adds launcher overhead for nothing.
- `dist.init_process_group(backend="nccl")` requires no address/port arguments — SageMaker
  populates the rendezvous environment variables.
- `torch.cuda.set_device(local_rank)` must be called **before** moving the model to the device,
  otherwise every rank lands on GPU 0.
- `NCCL_DEBUG=INFO` and `TORCH_DISTRIBUTED_DEBUG=DETAIL` in the estimator's `environment` make
  multi-node hangs diagnosable from CloudWatch logs.
- Only rank 0 should write to `SM_MODEL_DIR`; concurrent writes from all ranks corrupt the
  checkpoint.

## Instance Selection

Start at `ml.g5.xlarge` (one A10G) and scale up only when profiling shows the GPU is saturated.
`ml.g5.2xlarge` is the usual default for CV training; reach for `ml.p4d`/`ml.p5` only for large
models or genuine multi-node work. `max_run` caps runaway jobs — set it deliberately rather than
accepting the 24-hour default silently. `volume_size` must exceed the size of all input channels
combined, since SageMaker downloads them to the instance's EBS volume before the script starts.

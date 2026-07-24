# GPU Runners

Scope: targeting self-hosted GPU runners from a workflow and verifying the CUDA device
before running training jobs.

## Targeting a GPU Runner

Label self-hosted runners by capability (`self-hosted`, `gpu`, `linux`, or a GPU type
like `a100`), then target them and verify the device:

```yaml
jobs:
  train:
    runs-on: [self-hosted, gpu, linux]
    steps:
      - name: Verify GPU
        run: nvidia-smi
      - name: Set CUDA device
        run: echo "CUDA_VISIBLE_DEVICES=0" >> $GITHUB_ENV
```

## Notes

- Always set `timeout-minutes` on GPU jobs so a hung training run does not occupy the
  runner indefinitely.
- Pair a GPU runner label with a device flag through a matrix `include` entry when the
  same test suite must run on both CPU and CUDA.
- A GPU smoke-training job belongs on merge-to-main or `workflow_dispatch`, not on every
  push — it is the slowest tier.

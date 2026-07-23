# Distributed Training with Lightning

Multi-GPU and large-model training strategies: DDP and FSDP.

## Multi-GPU with DDP

```python
trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy="ddp",
    precision="16-mixed",
    sync_batchnorm=True,  # Important for multi-GPU with batch norm
)
```

## FSDP for Large Models

```python
from lightning.pytorch.strategies import FSDPStrategy

strategy = FSDPStrategy(
    sharding_strategy="FULL_SHARD",
    activation_checkpointing_policy={nn.TransformerEncoderLayer},
)

trainer = L.Trainer(
    accelerator="gpu",
    devices=4,
    strategy=strategy,
    precision="16-mixed",
)
```

## Notes

- `strategy="auto"` lets Lightning pick DDP/FSDP based on the detected hardware; name the strategy explicitly only when you need to override that choice.
- Never manually sync metrics in DDP — use `torchmetrics`, which handles cross-rank reduction automatically.
- Put data downloads in `prepare_data()`, which runs on rank 0 only; `setup()` runs on every rank and will race on a shared filesystem.
- `sync_batchnorm=True` converts BatchNorm layers to their synchronized variant, which matters when per-GPU batch size is small.

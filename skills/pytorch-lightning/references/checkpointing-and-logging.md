# Checkpointing and Logging

Saving and restoring checkpoints, attaching one or more loggers, and logging metrics and artifacts correctly.

## Checkpointing

`ModelCheckpoint` is the callback that writes checkpoints; combine it with `save_hyperparameters()` in the module so `load_from_checkpoint` can reconstruct the model without arguments.

```python
from lightning.pytorch.callbacks import ModelCheckpoint

ModelCheckpoint(
    dirpath="checkpoints/",
    filename="{epoch}-{val/acc:.3f}",
    monitor="val/acc",
    mode="max",
    save_top_k=3,
    save_last=True,
)
```

Restore for testing with `trainer.test(model, datamodule=datamodule, ckpt_path="best")`, or load a
module directly:

```python
model = ImageClassifier.load_from_checkpoint("checkpoints/last.ckpt")
assert model.hparams.num_classes == 10
```

## Multiple loggers at once

```python
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger, WandbLogger

loggers = [
    WandbLogger(project="my-project", name="experiment-1"),
    TensorBoardLogger(save_dir="logs/", name="my-project"),
    CSVLogger(save_dir="logs/", name="csv-logs"),
]

trainer = L.Trainer(logger=loggers)
```

## Logging Best Practices

```python
# Inside LightningModule methods:
self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
self.log_dict({"val/loss": loss, "val/acc": acc, "val/f1": f1}, on_epoch=True)

# Image/artifact logging goes through the logger's experiment handle:
if self.logger:
    self.logger.experiment.log({"images": wandb_images})
```

Always namespace metric keys by phase (`train/`, `val/`, `test/`) so the logger groups them, and
pass `torchmetrics` objects to `self.log` rather than pre-computed floats so epoch-level reduction
and DDP sync happen automatically.

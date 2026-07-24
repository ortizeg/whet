# Testing Lightning Code

Smoke tests and round-trip checks for LightningModules without running full training.

## Testing Lightning Code

Use `fast_dev_run=True` for smoke tests and `load_from_checkpoint` to verify hparams round-trip.

```python
import lightning as L
import torch

from my_project.models.classifier import ImageClassifier


def test_forward_shape() -> None:
    model = ImageClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    output = model(torch.randn(4, 3, 224, 224))
    assert output.shape == (4, 10)


def test_fast_dev_run() -> None:
    model = ImageClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, 10, (4,)))
    L.Trainer(fast_dev_run=True).fit(model, train_dataloaders=[batch])


def test_save_and_load(tmp_path) -> None:
    model = ImageClassifier(num_classes=10, backbone="resnet18", pretrained=False)
    path = tmp_path / "model.ckpt"
    trainer = L.Trainer(default_root_dir=tmp_path, max_epochs=0)
    trainer.strategy.connect(model)
    trainer.save_checkpoint(path)
    assert ImageClassifier.load_from_checkpoint(path).hparams.num_classes == 10
```

## Notes

- `fast_dev_run=True` runs a single train/val batch through the whole Trainer machinery — the cheapest way to catch shape, device, and logging errors.
- Always instantiate test models with `pretrained=False` and a tiny backbone (`resnet18`) so tests do not download weights.
- The checkpoint round-trip test is what proves `save_hyperparameters()` was called correctly.

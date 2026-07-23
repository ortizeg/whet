# Lightning and Experiment-Tracking Integration

Scope: wiring a composed Hydra config into PyTorch Lightning and logging the fully resolved config to an experiment tracker.

## Integration with Lightning

Hydra integrates cleanly with PyTorch Lightning by passing validated config objects into Lightning modules and data modules.

```python
import pytorch_lightning as pl
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = hydra_to_pydantic(cfg)

    model = MyModel(config.model)
    datamodule = MyDataModule(config.data)

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=config.training.gradient_clip_val,
        default_root_dir=cfg.output_dir,
    )
    trainer.fit(model, datamodule)
```

## Integration with experiment tracking

Log the resolved Hydra config to your experiment tracker so every run is fully reproducible.

```python
from omegaconf import OmegaConf
import wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Convert to dict for logging
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Log to W&B
    wandb.init(
        project=cfg.experiment_name,
        config=config_dict,
    )

    # Also save as YAML artifact
    with open("config_resolved.yaml", "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    wandb.save("config_resolved.yaml")
```

Always log or save the *resolved* config (`resolve=True`), not the raw one — otherwise interpolations like `${training.epochs}` are stored unresolved and the record cannot be replayed.

# W&B Hyperparameter Sweeps

Scope: defining sweep configurations and running sweep agents for Bayesian, grid, or random hyperparameter search.

W&B sweeps automate hyperparameter search using Bayesian optimization, grid search, or random search.

## Sweep Configuration

```python
import wandb

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/mAP", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2},
        "batch_size": {"values": [8, 16, 32, 64]},
        "optimizer": {"values": ["Adam", "AdamW", "SGD"]},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.5},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 10,
        "eta": 3,
    },
}

sweep_id = wandb.sweep(sweep_config, project="my-cv-project")
```

## Running a Sweep

```python
import wandb

def train_sweep() -> None:
    """Training function for sweep agent."""
    run = wandb.init()
    config = wandb.config

    model = build_model(config.dropout)
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.weight_decay)

    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_map = evaluate(model, val_loader)
        wandb.log({"train/loss": train_loss, "val/mAP": val_map, "epoch": epoch})

# Launch sweep agent
wandb.agent(sweep_id, function=train_sweep, count=50)
```

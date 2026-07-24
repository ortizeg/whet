# Artifact Storage

Scope: logging files, directories, matplotlib figures, dictionaries, and tables as MLflow run artifacts.

## Logging files

```python
import mlflow

# Log a single file
mlflow.log_artifact("checkpoints/best_model.pt", artifact_path="models")

# Log an entire directory
mlflow.log_artifacts("outputs/predictions/", artifact_path="predictions")

# Log a text file
with open("training_summary.txt", "w") as f:
    f.write(f"Best mAP: {best_map:.4f}\nBest epoch: {best_epoch}")
mlflow.log_artifact("training_summary.txt")
```

## Logging figures

```python
import mlflow
import matplotlib.pyplot as plt

# Create a training curve plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_losses, label="Train Loss")
ax.plot(val_losses, label="Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.set_title("Training Curves")

# Log the figure
mlflow.log_figure(fig, "plots/training_curves.png")
plt.close(fig)
```

## Logging dictionaries and tables

```python
import mlflow

# Log a dictionary as JSON
results = {
    "mAP": 0.45,
    "mAP50": 0.62,
    "mAP75": 0.38,
    "per_class": {"car": 0.52, "person": 0.48, "bike": 0.35},
}
mlflow.log_dict(results, "results/eval_metrics.json")

# Log a table
table = {
    "columns": ["class", "AP", "AP50", "AP75"],
    "data": [
        ["car", 0.52, 0.71, 0.45],
        ["person", 0.48, 0.65, 0.40],
        ["bike", 0.35, 0.50, 0.28],
    ],
}
mlflow.log_table(data=table, artifact_file="results/per_class_ap.json")
```

Use `artifact_path` to namespace artifacts (`models/`, `plots/`, `predictions/`) so
the run's artifact browser stays navigable, and always close matplotlib figures
after logging to avoid leaking memory during long training loops.

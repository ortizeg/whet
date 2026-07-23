# Hyperparameter Tuning

Bayesian hyperparameter search over a SageMaker estimator, with metrics scraped from training logs.

## Tuner Definition

Bayesian search over parameter ranges. Metrics are parsed from training logs via regex `metric_definitions`.

```python
from sagemaker.tuner import (
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
    IntegerParameter,
)


def create_tuner(estimator: PyTorch) -> HyperparameterTuner:
    return HyperparameterTuner(
        estimator=estimator,
        objective_metric_name="validation:accuracy",
        objective_type="Maximize",
        hyperparameter_ranges={
            "learning-rate": ContinuousParameter(1e-5, 1e-2, scaling_type="Logarithmic"),
            "batch-size": CategoricalParameter([16, 32, 64, 128]),
            "epochs": IntegerParameter(10, 100),
        },
        metric_definitions=[
            {"Name": "validation:accuracy", "Regex": r"val_acc=(\S+)"},
            {"Name": "validation:loss", "Regex": r"val_loss=(\S+)"},
        ],
        max_jobs=20,
        max_parallel_jobs=4,
        strategy="Bayesian",
    )
```

Launch it exactly like an estimator: `tuner.fit({"train": train_uri, "validation": val_uri})`.
`tuner.best_estimator()` returns the winning job for deployment or registration.

## Rules

- **Metrics come from stdout, not from an API.** The training script must actually print lines
  matching `metric_definitions` regexes — e.g. `logger.info("val_acc={}", acc)`. If nothing
  matches, every job reports no objective and the search degenerates to random.
- **The regex capture group is the value.** `r"val_acc=(\S+)"` captures everything up to
  whitespace; make the surrounding text distinctive enough not to match training-loss lines.
- **Hyperparameter keys must match the estimator's flags** (`learning-rate`, not
  `learning_rate`) — they are passed to the script the same way as static hyperparameters.
- **Use `scaling_type="Logarithmic"` for learning rates** and other quantities that span orders
  of magnitude; linear scaling wastes most of the budget in the uninteresting upper decade.
- **`max_parallel_jobs` trades speed against search quality.** Bayesian optimization learns from
  completed jobs, so running everything in parallel is equivalent to random search. Keep it to
  roughly a fifth of `max_jobs`.
- **`epochs` as a tuned parameter is usually wasteful** — prefer a fixed epoch budget with early
  stopping, and spend the search budget on learning rate, weight decay, and augmentation strength.
- Every trial is a full training job with its own instances. Cost is `max_jobs` × job cost —
  size the instance type down before scaling the search up.

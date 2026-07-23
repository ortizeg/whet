# PyTorch Lightning Skill

The PyTorch Lightning Skill standardizes model training across all CV/ML projects by enforcing consistent patterns for LightningModule, LightningDataModule, Trainer configuration, callbacks, and logging. It eliminates boilerplate training loops and ensures that every model in the repository follows the same lifecycle -- from data loading through training, validation, testing, and prediction. All module and data parameters are defined through Pydantic configurations, guaranteeing type safety and validation at initialization time rather than mid-training.

This skill covers the full Lightning ecosystem: writing clean `LightningModule` subclasses with explicit `training_step`, `validation_step`, and `configure_optimizers` methods; building `LightningDataModule` classes that encapsulate dataset preparation, splitting, and dataloader construction; configuring the `Trainer` with appropriate accelerators, strategies, precision settings, and checkpointing; and integrating callbacks for early stopping, learning rate monitoring, and model checkpointing.

## When to Use

- When building any model training pipeline, regardless of task (classification, detection, segmentation).
- When you need reproducible training with consistent logging and checkpointing behavior.
- When transitioning a raw PyTorch prototype into a production-ready training setup.
- When configuring distributed training across multiple GPUs or nodes.

## Key Features

- **LightningModule patterns** -- standardized structure for forward passes, loss computation, metric logging, and optimizer configuration.
- **LightningDataModule patterns** -- encapsulated data pipelines with `prepare_data`, `setup`, and dataloader methods separated by stage.
- **Pydantic-based configs** -- every hyperparameter, data path, and trainer setting is validated through frozen Pydantic models before training starts.
- **Callback library** -- reusable callbacks for early stopping, learning rate scheduling, gradient clipping monitoring, and custom metric tracking.
- **Logger integration** -- consistent patterns for TensorBoard, MLflow, and Weights & Biases logging through Lightning's logger abstraction.
- **Trainer configuration** -- opinionated defaults for precision, gradient accumulation, checkpointing strategies, and profiling.

## Related Skills

- **[Pydantic](../pydantic/)** -- provides the configuration validation layer that all Lightning module and data parameters must use.
- **[Abstraction Patterns](../abstraction-patterns/)** -- defines the ABC/Protocol interfaces that LightningModule subclasses implement for task-specific models.
- **[Docker CV](../docker-cv/)** -- builds training containers that include the correct CUDA runtime and Lightning dependencies.
- **[Code Quality](../code-quality/)** -- enforces type annotations on all Lightning hooks and callback methods through mypy strict mode.

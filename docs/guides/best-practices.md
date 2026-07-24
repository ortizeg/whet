# Best Practices

Comprehensive best practices for AI/CV projects using the whet framework.

## Project Organization

### Source Layout

Always use the `src/` layout to prevent import confusion:

```
my-project/
├── src/my_package/     # ✅ src layout
│   ├── __init__.py
│   └── module.py
└── tests/
    └── test_module.py
```

### Configuration Management

- Use Hydra for experiment configs (composable YAML)
- Use Pydantic for runtime validation of all configs
- Never hardcode paths, URLs, or credentials
- Store defaults in config files, override via CLI or environment

### Test Structure

```
tests/
├── unit/           # Fast, isolated (milliseconds per test)
├── integration/    # Multi-component (seconds per test)
├── e2e/            # Full pipeline (minutes per test)
└── conftest.py     # Shared fixtures
```

## Code Quality

### Type Hints Everywhere

```python
# ✅ Full type hints with return type
def process_image(
    image: np.ndarray,
    *,
    resize: tuple[int, int] | None = None,
) -> torch.Tensor:
    ...

# ❌ Missing types
def process_image(image, resize=None):
    ...
```

### Pydantic for All Configs

```python
# ✅ Validated, typed, documented
class TrainingConfig(BaseModel):
    lr: float = Field(gt=0, default=1e-3, description="Learning rate")
    epochs: int = Field(ge=1, default=100)

# ❌ Unvalidated dict
config = {"lr": 0.001, "epochs": 100}
```

### Abstraction Patterns

- Wrap external libraries (cv2, PIL) behind your own interfaces
- Use ABC or Protocol for interchangeable components
- Don't abstract standard library or stable core frameworks prematurely

## Training Best Practices

### Reproducibility

```python
import lightning as L

L.seed_everything(42, workers=True)

trainer = L.Trainer(
    deterministic=True,  # Deterministic algorithms
)
```

### Checkpointing

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    monitor="val/mAP",
    mode="max",
    save_top_k=3,
    save_last=True,
    filename="{epoch}-{val_mAP:.4f}",
)
```

### Logging

```python
# ✅ Use structured logging
import logging
logger = logging.getLogger(__name__)
logger.info("Training started", extra={"epochs": 100, "lr": 1e-3})

# ❌ Print statements
print(f"Training started with lr={lr}")
```

## Data Management

- **Object storage plus a versioned manifest** for large datasets and model weights -- keep the bulk data out of Git
- **Pydantic** for validating data schemas
- **Explicit splits** -- never let data leak between train/val/test
- **Data validation** -- check for corrupted images, missing labels, class imbalance

## CI/CD

- **Code Review agent** (blocking) -- lint, format, type check on every PR
- **Test Engineer agent** (blocking) -- tests + 80% coverage on every PR
- **Pre-commit hooks** -- catch issues before they reach CI
- **Training validation** -- smoke test training on merge to main

## Deployment

- **ONNX export** for production inference (faster, portable)
- **Docker** with multi-stage builds (slim inference images)
- **Health checks** on all serving endpoints
- **Pydantic schemas** for API request/response validation

## Team Workflows

- Use PRs for all changes (no direct commits to main)
- Code Review agent blocks merge on quality violations
- Document architecture decisions in code comments or ADRs
- Share workspace settings via `.vscode/` (committed to git)

## Common Mistakes to Avoid

1. **Using `Any` without documentation** -- always explain why `Any` is necessary
2. **Skipping type hints** -- mypy strict mode catches bugs early
3. **Using dicts for configs** -- Pydantic provides validation and documentation
4. **Direct library usage** -- wrap cv2, PIL, etc. for testability
5. **Print statements** -- use `logging` module instead
6. **Hardcoded paths** -- use config files or environment variables
7. **No validation loop** -- always validate during training
8. **Missing gradient clipping** -- prevents gradient explosion
9. **Wrong normalization** -- use ImageNet stats for pretrained models
10. **Global state** -- makes testing and debugging much harder
11. **Ignoring CI failures** -- fix immediately, don't bypass
12. **Large commits** -- small, focused PRs are easier to review
13. **No reproducibility** -- set seeds, log configs, version data
14. **Skipping tests** -- fix or remove, never skip permanently

# Expert Coder Agent

The Expert Coder is the primary coding assistant for all development tasks in AI/CV projects. It guides code generation to follow project standards consistently.

## Purpose

Every piece of code generated follows these principles:

- **Abstraction First** -- wrap external libraries (cv2, PIL, ffmpeg) behind clean interfaces
- **Pydantic Everywhere** -- use `BaseModel` for all configs (Level 1) and data structures (Level 2)
- **Type Safety** -- full type hints on all functions, no `Any` unless documented
- **Testability** -- dependency injection, pure functions, no global state

## Strictness Level

**Advisory** -- suggests and guides but does not block. Actual enforcement happens via pre-commit hooks and the Code Review agent in CI.

## How to Use

The Expert Coder reads the relevant skill files and generates code following all patterns:

```
You: "Create a DataModule for COCO object detection"

Expert Coder:
1. Reads pytorch-lightning, pydantic, abstraction-patterns skills
2. Creates DataConfig with Pydantic validation
3. Implements LightningDataModule with proper hooks
4. Wraps data loading in abstractions
5. Adds full type hints and docstrings
```

## Patterns Enforced

### Correct: Pydantic Config + Abstraction

```python
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    data_dir: str
    batch_size: int = Field(ge=1, default=32)
    num_workers: int = Field(ge=0, default=4)
```

### Incorrect: Raw Dict + Direct Library Use

```python
# ❌ No validation, no types
config = {"data_dir": "/data", "batch_size": 32}
cap = cv2.VideoCapture("video.mp4")  # Direct library use
```

## What It Provides

- Structure guidance for module organization
- Pattern enforcement for abstractions and Pydantic usage
- Type safety with comprehensive type hints
- Documentation with helpful docstrings
- ML/CV-specific best practices

## Related Skills

The Expert Coder reads: `pydantic`, `abstraction-patterns`, `code-quality`, `pytorch-lightning`, and others as needed.

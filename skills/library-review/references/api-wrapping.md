# Wrapping and Migrating Third-Party APIs

Scope: isolating an adopted library behind a project-owned interface, the full
ExperimentTracker wrapper example including the Null Object pattern, and the checklist for
swapping one library for another.

## Contents

- [Why Wrap](#why-wrap)
- [Wrapping Pattern for ML Libraries](#wrapping-pattern-for-ml-libraries)
- [Migration Checklist](#migration-checklist)

## Why Wrap

Direct calls scattered across files (`some_library.process(...)`) mean every file
changes when the library's API changes. Wrapped behind a Protocol, only the wrapper
updates:

```python
from typing import Protocol

class ImageProcessor(Protocol):
    """Interface for image processing."""
    def process(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...

class SomeLibraryProcessor:
    """Image processor using some_library."""

    def process(self, image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return some_library.process(image, mode="fast", threshold=threshold)
```

## Wrapping Pattern for ML Libraries

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class ExperimentTracker(ABC):
    """Abstract interface for experiment tracking."""

    @abstractmethod
    def log_metric(self, name: str, value: float, step: int) -> None: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def log_artifact(self, path: str) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


class WandbTracker(ExperimentTracker):
    """W&B implementation of experiment tracker."""

    def __init__(self, project: str, config: dict[str, Any]) -> None:
        import wandb
        self.run = wandb.init(project=project, config=config)

    def log_metric(self, name: str, value: float, step: int) -> None:
        import wandb
        wandb.log({name: value}, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        import wandb
        wandb.config.update(params)

    def log_artifact(self, path: str) -> None:
        import wandb
        artifact = wandb.Artifact("output", type="result")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        import wandb
        wandb.finish()


# An MLflowTracker implements the same four methods against mlflow.log_metric /
# log_params / log_artifact / end_run — swapping the backend is a wrapper-only change.


class NullTracker(ExperimentTracker):
    """No-op tracker for when tracking is disabled."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def finish(self) -> None:
        pass
```

## Migration Checklist

When a library must be replaced, the wrapping strategy makes migration manageable:

1. **Identify all usage points** by searching for imports of the old library.
2. **Evaluate the replacement** using the full evaluation checklist.
3. **Create the new wrapper** implementing the same interface.
4. **Write comparison tests** ensuring the new library produces equivalent results.
5. **Migrate in phases**: start with non-critical code paths.
6. **Run benchmarks** to verify performance is acceptable.
7. **Update documentation** and dependency specifications.
8. **Remove the old dependency** once migration is complete.

```bash
# Find all imports of the old library
grep -rn "import old_library" src/ tests/
grep -rn "from old_library" src/ tests/
```

---
name: library-review
description: >
  Use this skill when deciding whether to adopt a third-party Python library — evaluating
  maintenance, community, docs, type-hint support, license compatibility, security
  history, and performance, then classifying it Adopt/Trial/Assess/Hold. Reach for it any
  time you'd otherwise add a dependency on gut feel or wonder "is this package safe and
  maintained enough to depend on?", even if the user doesn't say "library review". For
  designing wrappers around a library once adopted, see abstraction-patterns.
---

# Library Review and Evaluation Framework

Every dependency is a long-term commitment carrying security, maintenance, breaking-change, license, and transitive-dependency risk. Evaluate new libraries before adoption using a technology radar (Adopt / Trial / Assess / Hold) plus a wrapping strategy that isolates third-party APIs behind project-owned interfaces.

## Evaluation Criteria Checklist

### 1. Maintenance Status

Check whether the library is actively maintained.

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| Last commit | < 3 months | 3-12 months | > 12 months |
| Last release | < 6 months | 6-18 months | > 18 months |
| Open issues | Triaged, responded to | Growing, some responses | Ignored |
| CI status | Passing | Flaky | Failing or none |
| Python version support | Current + previous | Current only | Outdated |

```bash
# Check last commit date
gh api repos/OWNER/REPO --jq '.pushed_at'

# Check latest release
gh api repos/OWNER/REPO/releases/latest --jq '.published_at'

# Check open issues count
gh api repos/OWNER/REPO --jq '.open_issues_count'
```

### 2. Community Size

A larger community means more eyes on bugs, more documentation, and more Stack Overflow answers.

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| GitHub stars | > 1000 | 100-1000 | < 100 |
| Contributors | > 20 | 5-20 | < 5 |
| PyPI downloads/month | > 100K | 10K-100K | < 10K |
| Stack Overflow questions | > 100 | 10-100 | < 10 |

```bash
# Check GitHub stars and contributors
gh api repos/OWNER/REPO --jq '{stars: .stargazers_count, forks: .forks_count}'

# Check PyPI download stats
pip install pypistats
pypistats recent PACKAGE_NAME
```

### 3. Documentation Quality

Poor documentation is a reliable signal of a library that will be difficult to use and maintain.

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| API reference | Complete, auto-generated | Partial | Missing |
| Tutorials/guides | Multiple, up-to-date | Basic README only | Outdated or missing |
| Examples | Runnable, tested | Untested snippets | None |
| Changelog | Detailed, per-version | Brief summaries | Missing |
| Migration guides | Provided for breaking changes | Partial | None |

### 4. Type Hint Support

For a project using strict type checking, type hint support is a requirement rather than a nice-to-have.

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| Inline type hints | Full, PEP 484+ | Partial | None |
| py.typed marker | Present | Missing but stubs available | Neither |
| MyPy compatibility | Passes strict mode | Passes basic mode | Fails |
| Type stubs (typeshed) | Official stubs available | Community stubs | None |

```python
# Check for py.typed marker
import importlib.resources
try:
    importlib.resources.files("library_name").joinpath("py.typed")
    print("py.typed marker found")
except (TypeError, FileNotFoundError):
    print("No py.typed marker")
```

### 5. License Compatibility

Verify the license is compatible with your project's license.

| License | Compatible with MIT | Compatible with Apache 2.0 | Notes |
|---------|--------------------|----|-------|
| MIT | Yes | Yes | Most permissive |
| Apache 2.0 | Yes | Yes | Patent grant |
| BSD 2/3 | Yes | Yes | Permissive |
| LGPL | Yes (dynamic linking) | Yes (dynamic linking) | Must not modify |
| GPL | No | No | Copyleft, viral |
| AGPL | No | No | Copyleft, network use |

```bash
# Check license via GitHub API
gh api repos/OWNER/REPO --jq '.license.spdx_id'

# Check license via pip
pip show PACKAGE_NAME | grep License
```

### 6. Security History

Review the library's security track record.

```bash
# Check for known vulnerabilities
pip-audit --requirement requirements.txt

# Check GitHub security advisories
gh api repos/OWNER/REPO/security-advisories --jq '.[].summary'

# Check the Safety database
safety check --full-report
```

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| CVE history | None or quickly patched | Few, patched within weeks | Multiple, slow patches |
| Dependency audit | Clean | Minor issues | Critical vulnerabilities |
| Security policy | Published SECURITY.md | Informal process | None |

### 7. Performance Benchmarks

For performance-sensitive libraries (image processing, inference, data loading):

```python
import time
import numpy as np

def benchmark_library(func, input_data, num_runs: int = 100) -> dict[str, float]:
    """Benchmark a library function."""
    # Warmup
    for _ in range(10):
        func(input_data)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        func(input_data)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "p95_ms": np.percentile(times, 95),
    }

# Compare candidates
results_a = benchmark_library(library_a.process, test_data)
results_b = benchmark_library(library_b.process, test_data)
```

### 8. Integration with Existing Stack

Verify the library works with your existing tools and dependencies.

| Question | Answer |
|----------|--------|
| Does it work with your Python version? | Check `python_requires` in setup.cfg/pyproject.toml |
| Does it conflict with existing dependencies? | Run `pip check` after install |
| Does it support your OS/platform? | Check CI matrix and platform wheels |
| Does it integrate with PyTorch/Lightning? | Check for official integrations |
| Can it be installed with uv/conda? | Check PyPI and conda-forge availability |

```bash
# Check for dependency conflicts
pip install CANDIDATE_LIBRARY
pip check

# Check conda-forge availability
conda search -c conda-forge CANDIDATE_LIBRARY
```

## Decision Framework

Use the Technology Radar approach to classify libraries:

### Adopt

The library is proven and recommended for use across the project.

**Criteria:**
- All checklist items are green or yellow.
- Used in production by the team or well-known organizations.
- Active maintenance with responsive maintainers.
- Good type support and documentation.
- Compatible license.

**Examples:** PyTorch, Pydantic, Ruff, pytest, Pillow, NumPy.

### Trial

The library shows promise and should be tested in a non-critical part of the project.

**Criteria:**
- Most checklist items are green or yellow.
- Relatively new but gaining traction quickly.
- Team member willing to champion and maintain the integration.
- Wrapped behind a project interface to limit exposure.

**Examples:** A new image augmentation library, a faster data loader, a specialized metric library.

### Assess

The library is interesting but needs further evaluation before any use.

**Criteria:**
- Mixed signals on the checklist.
- No team member has hands-on experience.
- Potential overlap with existing tools.

**Action:** Assign a team member to create a proof-of-concept in a branch.

### Hold

The library should not be adopted at this time.

**Criteria:**
- Multiple red signals on the checklist.
- Abandoned or poorly maintained.
- License incompatibility.
- Better alternatives exist.
- Significant security concerns.

## Wrapping Strategy

Always wrap third-party APIs behind project-owned interfaces. This isolates the project from breaking changes and makes migration possible.

### Why Wrap

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

### Wrapping Pattern for ML Libraries

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

## Migration Planning

When a library must be replaced, the wrapping strategy makes migration manageable:

### Migration Checklist

1. **Identify all usage points** by searching for imports of the old library.
2. **Evaluate the replacement** using the full evaluation checklist above.
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

## Review Template

Use this template when proposing a new dependency:

```markdown
## Library Review: [Library Name]

### Basic Information
- **Name**: [library-name]
- **Version**: [x.y.z]
- **License**: [MIT/Apache/etc.]
- **PyPI**: [link]
- **Repository**: [link]
- **Documentation**: [link]

### Purpose
[Why do we need this library? What problem does it solve?]

### Alternatives Considered
| Library | Pros | Cons |
|---------|------|------|

### Evaluation Checklist
Score each criterion above (maintenance, community, docs, types, license,
security, performance, integration) green/yellow/red.

### Decision
- [ ] Adopt  - [ ] Trial  - [ ] Assess  - [ ] Hold

### Wrapping Plan
[How will this library be wrapped behind a project interface?]

### Reviewer
[Name, Date]
```

## Automated Checks

Integrate dependency review into CI:

```yaml
# .github/workflows/dependency-review.yml
name: Dependency Review

on:
  pull_request:

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Dependency Review
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high
          deny-licenses: GPL-3.0, AGPL-3.0

      - name: Audit dependencies
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.txt
```

## Regular Review Schedule

Dependencies should be reviewed periodically, not just at adoption:

```bash
# Monthly: Check for outdated packages
pip list --outdated

# Monthly: Run security audit
pip-audit

# Quarterly: Review dependency tree for unused packages
pipdeptree --warn silence | grep -E "^\w"

# Annually: Full evaluation of all dependencies against the checklist
```

## Best Practices

1. **Evaluate before installing**: Run through the checklist before adding any dependency.
2. **Wrap all third-party APIs**: Isolate external code behind project interfaces.
3. **Minimize transitive dependencies**: Prefer libraries with few dependencies of their own.
4. **Pin versions**: Use exact versions in lock files, compatible ranges in requirements.
5. **Audit regularly**: Run `pip-audit` in CI and review outdated packages monthly.
6. **Document decisions**: Use the review template for every new dependency.
7. **Use the Null Object pattern**: Provide no-op implementations for optional dependencies.
8. **Prefer standard library**: Use built-in modules when they are sufficient.
9. **Track the Technology Radar**: Maintain a team document classifying all dependencies.
10. **Plan for migration**: Assume every library will eventually need to be replaced.


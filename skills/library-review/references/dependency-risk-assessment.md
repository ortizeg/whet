# Dependency Risk Assessment

Scope: the green/yellow/red scoring tables and the commands that gather the evidence for
maintenance, community, documentation, type-hint support, performance, and stack
integration. License and security scoring live in the security-and-license review.

## Contents

- [1. Maintenance Status](#1-maintenance-status)
- [2. Community Size](#2-community-size)
- [3. Documentation Quality](#3-documentation-quality)
- [4. Type Hint Support](#4-type-hint-support)
- [7. Performance Benchmarks](#7-performance-benchmarks)
- [8. Integration with Existing Stack](#8-integration-with-existing-stack)

## 1. Maintenance Status

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

## 2. Community Size

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

## 3. Documentation Quality

Poor documentation is a reliable signal of a library that will be difficult to use and maintain.

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| API reference | Complete, auto-generated | Partial | Missing |
| Tutorials/guides | Multiple, up-to-date | Basic README only | Outdated or missing |
| Examples | Runnable, tested | Untested snippets | None |
| Changelog | Detailed, per-version | Brief summaries | Missing |
| Migration guides | Provided for breaking changes | Partial | None |

## 4. Type Hint Support

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

## 7. Performance Benchmarks

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

## 8. Integration with Existing Stack

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

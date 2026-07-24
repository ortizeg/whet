# Coverage Configuration and CI Integration

Pytest and coverage settings in `pyproject.toml`, useful invocations, and the GitHub Actions test workflow.

## Coverage Configuration

Configure coverage in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: slow tests (deselect with -m 'not slow')", "gpu: requires GPU"]
addopts = ["--strict-markers", "-ra", "--tb=short"]

[tool.coverage.run]
source = ["src/myproject"]
omit = ["*/tests/*", "*/__pycache__/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:", "raise NotImplementedError"]
```

Common invocations: `pytest --cov=src/myproject --cov-report=term-missing` (coverage), `-m "not slow"` (skip slow), `tests/unit/test_model.py` (one file), `-k "test_bbox"` (by pattern).

## CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src/myproject --cov-report=xml -m "not slow"
      - uses: codecov/codecov-action@v4
        with:
          file: coverage.xml
```

## Coverage Requirements

- **Overall:** minimum 80% line coverage
- **New code:** at least 90%
- **Critical paths:** 100% — model forward pass, data loading, config validation
- **No skipped tests:** fix or delete them; a permanently skipped test is a lie

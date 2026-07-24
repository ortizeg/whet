# Pixi Task Definitions

Scope: defining project commands in `pixi.toml` — simple tasks, task dependency chains,
environment variables, and feature-scoped tasks.

Tasks replace Makefiles and shell scripts. They are defined in pixi.toml and run with
`pixi run <task>`.

## Simple Tasks

```toml
[tasks]
test = "pytest tests/ -v"
lint = "ruff check ."
format = "ruff format ."
typecheck = "mypy src/ --strict"
```

## Tasks with Dependencies

Tasks can depend on other tasks. Pixi runs dependencies first.

```toml
[tasks]
lint = "ruff check ."
format-check = "ruff format . --check"
typecheck = "mypy src/ --strict"
test = "pytest tests/ -v --cov=src --cov-fail-under=80"

# This runs lint, format-check, and typecheck before running tests
quality = { depends-on = ["lint", "format-check", "typecheck", "test"] }
```

## Tasks with Arguments and Environment Variables

```toml
[tasks]
train = { cmd = "python -m my_project.train", env = { WANDB_MODE = "online" } }
train-debug = { cmd = "python -m my_project.train", env = { WANDB_MODE = "disabled", CUDA_VISIBLE_DEVICES = "0" } }
serve = { cmd = "uvicorn my_project.api:app --host 0.0.0.0 --port 8000" }
```

## Feature-Specific Tasks

Tasks can be scoped to specific features, meaning they are only available when that
feature is activated.

```toml
[feature.docs.tasks]
docs-build = "mkdocs build"
docs-serve = "mkdocs serve"

[feature.onnx.tasks]
export-onnx = "python scripts/export_onnx.py"
validate-onnx = "python scripts/validate_onnx.py"
```

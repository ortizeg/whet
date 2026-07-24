---
name: vscode
description: >
  Use this skill when configuring VS Code for CV/ML development — workspace settings.json,
  Ruff and MyPy editor integration, Python debug launch configs, tasks, remote-SSH to GPU
  servers, dev containers, and recommended extensions. Reach for it any time you'd
  otherwise hand-edit .vscode files or set up a remote GPU dev environment, even if the
  user just says "set up my editor" or "debug this in VS Code". This is editor setup; the
  lint/type standards live in code-quality and commit hooks in pre-commit.
---

# VS Code Skill

Configure VS Code for productive computer vision and machine learning development with
integrated linting, debugging, and remote GPU server support. This page holds the core
workspace settings you need almost every time; the deep dives below cover launch configs,
tasks, remote SSH, dev containers, and the extension list.

**Scope boundary:** this is editor setup. The Ruff/MyPy rule sets themselves live in the
`code-quality` skill, and commit-time enforcement lives in `pre-commit`.

## The settings core

Point VS Code at the pixi interpreter, make Ruff the formatter, run MyPy strict, and hide
the big ML directories from search. This is the 80% of `.vscode/settings.json`:

```jsonc
{
    // Python — use the pixi-managed interpreter
    "python.defaultInterpreterPath": "${workspaceFolder}/.pixi/envs/default/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.diagnosticMode": "workspace",

    // Ruff (replaces black, isort, flake8)
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "ruff.lineLength": 100,

    // mypy
    "mypy-type-checker.args": [
        "--strict",
        "--config-file=${workspaceFolder}/pyproject.toml"
    ],

    // Testing
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests", "-v", "--tb=short"],
    "python.testing.unittestEnabled": false,

    // Keep ML output directories out of search
    "search.exclude": {
        "**/data": true,
        "**/checkpoints": true,
        "**/outputs": true,
        "**/wandb": true,
        "**/mlruns": true,
        "**/lightning_logs": true,
        "**/.pixi": true
    },

    // Editor
    "editor.rulers": [100],
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true
}
```

The full settings file — file associations, `files.exclude`, terminal PATH, and Jupyter
notebook formatting — is in `references/workspace-settings.md`.

## Workspace vs user settings

Workspace settings (interpreter path, Ruff config, exclusions) go in `.vscode/` and are
committed to git. User preferences (theme, font size, keybindings) stay in
`~/.config/Code/User/` and are never committed.

## Conventions

1. **Commit `.vscode/`** -- share workspace settings, launch configs, and extension recommendations
2. **Don't commit user settings** -- themes, fonts, and personal keybindings stay user-level
3. **Use pixi interpreter** -- ensures everyone uses the same Python and packages
4. **Configure search exclusions** -- hide data/, checkpoints/, and wandb/ from search
5. **Set up debug configs** -- pre-configure training, testing, and serving debug sessions
6. **Use tasks** -- define pixi run commands as VS Code tasks for quick access
7. **Install recommended extensions** -- prompt teammates on first open
8. **Configure remote SSH** -- port-forward TensorBoard and MLflow for remote GPU development
9. **Use Dev Containers** -- consistent GPU environment across team members
10. **Set rulers at line length** -- visual guide matching Ruff's `line-length = 100`

## Anti-patterns

- **A global system interpreter** — pointing `python.defaultInterpreterPath` at `/usr/bin/python` silently diverges the editor from the pixi environment CI uses.
- **Committing user settings** — themes, fonts, and keybindings in `.vscode/settings.json` fight every teammate's preferences.
- **Leaving `data/` and `checkpoints/` searchable** — a workspace search over a dataset directory hangs the editor.
- **Mixing formatters** — black or autopep8 alongside Ruff produces churn on every save; Ruff is the single formatter.
- **`"justMyCode": true` in ML debug configs** — you cannot step into Lightning, torch, or FastAPI, which is where the bug usually is.
- **Missing `PYTHONPATH` for src-layout** — debug launches fail to import the package unless `${workspaceFolder}/src` is on the path.
- **A dev container without `--shm-size`** — DataLoader workers die on Docker's 64 MB default shared memory.

## Deep dives

- `references/workspace-settings.md` — read when writing the complete `.vscode/settings.json`, wiring the pixi interpreter into the terminal, defining `tasks.json`, or deciding what belongs in workspace vs user scope.
- `references/debug-configs.md` — read when setting up `launch.json` to debug a training run, pytest, or a FastAPI serving endpoint.
- `references/remote-ssh-gpu.md` — read when developing over Remote-SSH on a GPU server or port-forwarding TensorBoard/MLflow.
- `references/devcontainers.md` — read when building a GPU-enabled dev container from the project Dockerfile.
- `references/extensions.md` — read when writing `.vscode/extensions.json` or deciding which extensions the project should recommend.

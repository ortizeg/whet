# Workspace Settings, Tasks, and Interpreter

Scope: the complete `.vscode/settings.json` for a CV/ML project, the pixi interpreter
wiring, `.vscode/tasks.json`, and the workspace-vs-user settings split.

## Contents

- [.vscode/settings.json](#vscodesettingsjson)
- [Python Interpreter with Pixi](#python-interpreter-with-pixi)
- [Task Definitions](#task-definitions)
- [Workspace vs User Settings](#workspace-vs-user-settings)

## .vscode/settings.json

```jsonc
{
    // Python
    "python.defaultInterpreterPath": "${workspaceFolder}/.pixi/envs/default/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.autoImportCompletions": true,

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
    "python.analysis.diagnosticMode": "workspace",

    // Testing
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v",
        "--tb=short"
    ],
    "python.testing.unittestEnabled": false,

    // File associations
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml",
        "Dockerfile*": "dockerfile",
        "*.toml": "toml"
    },

    // Exclude large directories from explorer and search
    "files.exclude": {
        "**/__pycache__": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/.pytest_cache": true,
        "**/htmlcov": true,
        "**/site": true,
        "**/*.egg-info": true
    },
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
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,

    // Terminal
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.env.linux": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    },

    // Jupyter
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "notebook.formatOnSave.enabled": true,
    "notebook.codeActionsOnSave": {
        "source.fixAll.ruff": "explicit"
    }
}
```

## Python Interpreter with Pixi

Configure VS Code to use the pixi-managed Python:

```jsonc
{
    // Auto-detect pixi environment
    "python.defaultInterpreterPath": "${workspaceFolder}/.pixi/envs/default/bin/python",

    // Alternative: use pixi run prefix for terminal commands
    "terminal.integrated.env.linux": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    },
    "terminal.integrated.env.osx": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    }
}
```

## Task Definitions

### .vscode/tasks.json

```jsonc
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Lint",
            "type": "shell",
            "command": "pixi run lint",
            "group": "test",
            "problemMatcher": ["$eslint-stylish"],
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Format",
            "type": "shell",
            "command": "pixi run format",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Type Check",
            "type": "shell",
            "command": "pixi run typecheck",
            "group": "test",
            "problemMatcher": ["$tsc"],
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Test",
            "type": "shell",
            "command": "pixi run test",
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Train",
            "type": "shell",
            "command": "pixi run python -m my_project.train",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        }
    ]
}
```

## Workspace vs User Settings

| Setting | Scope | Location |
|---------|-------|----------|
| Python interpreter path | Workspace | `.vscode/settings.json` |
| Ruff configuration | Workspace | `.vscode/settings.json` |
| File exclusions | Workspace | `.vscode/settings.json` |
| Theme, font size | User | `~/.config/Code/User/settings.json` |
| Keybindings | User | `~/.config/Code/User/keybindings.json` |
| Extension sync | User | VS Code settings sync |

**Rule:** Workspace settings go in `.vscode/` and are committed to git. User preferences
stay in your user settings and are never committed.

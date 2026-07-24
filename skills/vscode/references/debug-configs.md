# Debug Configurations

Scope: the full `.vscode/launch.json` with debugpy configurations for training, pytest,
FastAPI serving, and arbitrary scripts.

## Contents

- [.vscode/launch.json](#vscodelaunchjson)
- [Shared Config Notes](#shared-config-notes)

## .vscode/launch.json

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        // =====================================================================
        // Training — debug a Lightning training run
        // =====================================================================
        {
            "name": "Train: Debug",
            "type": "debugpy",
            "request": "launch",
            "module": "my_project.train",
            "args": [
                "trainer.max_epochs=2",
                "trainer.fast_dev_run=true",
                "data.batch_size=4"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Tests — debug pytest with current file or specific test
        // =====================================================================
        {
            "name": "Test: Current File",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "${file}",
                "-v",
                "--tb=short",
                "--no-header"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },
        {
            "name": "Test: All",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--tb=short"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Inference — debug FastAPI serving endpoint
        // =====================================================================
        {
            "name": "Serve: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "my_project.serve:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src",
                "MODEL_PATH": "${workspaceFolder}/models/best.onnx"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Script — debug any Python script
        // =====================================================================
        {
            "name": "Script: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        }
    ]
}
```

## Shared Config Notes

- `"justMyCode": false` lets you step into library code (Lightning, torch, FastAPI),
  which is where most CV/ML bugs actually surface.
- `"PYTHONPATH": "${workspaceFolder}/src"` is required for src-layout projects so the
  debugger resolves the package without an editable install.
- `"console": "integratedTerminal"` keeps `tqdm` progress bars and interactive prompts
  working; the default debug console mangles them.
- Set `CUDA_VISIBLE_DEVICES` in the training config to pin the debug run to one GPU.

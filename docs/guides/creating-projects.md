# Creating Projects

This guide walks through creating a new AI/CV project using the whet framework.

## Prerequisites

- [Claude Code](https://claude.ai/claude-code) installed
- This repository cloned locally
- `pixi` installed ([install guide](https://pixi.sh))

## Step 1: Start Claude Code

```bash
claude
```

## Step 2: Tell Claude What You Want

```
You: "Using whet, create a new pytorch-training-project
      called 'face-detection' for detecting faces in video streams"
```

Claude reads the Master Skill and begins the initialization flow.

## Step 3: Provide Project Details

Claude will ask for:

| Field | Example | Default |
|-------|---------|---------|
| Project name | Face Detection System | (required) |
| Project slug | face-detection-system | (auto-generated) |
| Package name | face_detection_system | (auto-generated) |
| Author | Enrique G. Ortiz | (required) |
| Email | ortizeg@gmail.com | (required) |
| Description | Real-time face detection | (required) |
| Python version | 3.11 | 3.11 |
| Version | 0.1.0 | 0.1.0 |

## Step 4: Choose an Archetype

| Archetype | Best For |
|-----------|----------|
| `pytorch-training-project` | Training models with Lightning + Hydra |
| `cv-inference-service` | Deploying models via FastAPI + ONNX |
| `research-notebook` | Jupyter-based experimentation |
| `library-package` | Reusable PyPI packages |
| `data-processing-pipeline` | Dataset ETL workflows |
| `model-zoo` | Pretrained model collections |

## Step 5: Select Optional Skills

Choose which experiment tracking and data management tools to include:

| Skill | Purpose |
|-------|---------|
| Weights & Biases | Cloud experiment tracking with rich media |
| MLflow | Self-hosted experiment tracking + model registry |
| TensorBoard | Local training visualization |

## Step 6: Project Generation

Claude generates the complete project structure:

```
face-detection-system/
├── .github/workflows/
│   ├── ci.yml
│   └── test.yml
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml
├── pyproject.toml
├── README.md
├── configs/
│   ├── config.yaml
│   ├── model/
│   └── data/
├── src/face_detection_system/
│   ├── __init__.py
│   ├── train.py
│   ├── models/
│   ├── data/
│   └── utils/
└── tests/
    ├── conftest.py
    └── test_model.py
```

## Step 7: Post-Initialization

After generation, set up the development environment:

```bash
cd face-detection-system

# Initialize git
git init
git add .
git commit -m "Initial project from pytorch-training-project archetype"

# Install dependencies
pixi install

# Install pre-commit hooks
uv run pre-commit install

# Verify everything works
uv run lint
uv run test
```

## Full Example

Creating a face detection project with W&B tracking:

```
You: "Create a pytorch-training-project called 'face-detection' with wandb"

Claude:
1. Reads master-skill/SKILL.md
2. Collects project details
3. Generates pytorch-training-project structure
4. Includes wandb integration in pixi.toml and training code
5. Sets up W&B logger in Lightning Trainer
6. Adds .env.example with WANDB_API_KEY placeholder

You: "Now implement a RetinaFace detector"

Claude:
1. Reads expert-coder, pytorch-lightning, pydantic skills
2. Creates ModelConfig with Pydantic
3. Implements RetinaFaceModule(pl.LightningModule)
4. Wraps backbone loading in abstraction
5. Adds comprehensive type hints and docstrings
```

## What Happens Behind the Scenes

The Master Skill orchestrates these skills during generation:

1. **code-quality** -- configures Ruff + mypy in pyproject.toml
2. **pixi** -- generates pixi.toml with all dependencies
3. **pre-commit** -- creates .pre-commit-config.yaml
4. **github-actions** -- generates CI workflows
5. **testing** -- sets up pytest configuration
6. **pytorch-lightning** -- adds Lightning patterns (if training archetype)
7. **wandb/mlflow/tensorboard** -- adds tracking integration (if selected)

Each skill contributes its specific configuration and code patterns to the generated project.

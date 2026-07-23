# Skills Overview

Skills are the core building blocks of the AI/CV Claude Skills framework. Each skill is a focused knowledge module that teaches Claude Code how to use a specific tool, library, or pattern correctly in the context of computer vision and deep learning projects.

## How Skills Work

Each skill lives in its own directory under `skills/` and contains:

- **`SKILL.md`** -- The full skill definition loaded by Claude Code. Contains coding conventions, templates, anti-patterns, and integration guidance.
- **`README.md`** -- Human-readable overview of the skill's purpose and scope.
- **`configs/`** -- Example configuration files relevant to the skill.
- **`examples/`** -- Sample code demonstrating the skill in action.

When you reference a skill in a Claude Code session, Claude reads the `SKILL.md` and uses its contents to guide code generation. The skill does not execute code -- it provides expert context that shapes Claude's output.

## Skill Categories

### Core Framework

These skills define the foundational patterns for all projects.

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [Master Skill](master-skill.md) | Universal coding conventions for all CV/ML code | Python, typing |
| [Pydantic](pydantic.md) | Strict data validation and configuration models | Pydantic v2 |
| [Code Quality](code-quality.md) | Linting, formatting, and type checking setup | ruff, mypy |
| [Loguru](loguru.md) | Structured logging for all projects (mandatory convention) | loguru |
| [Abstraction Patterns](abstraction-patterns.md) | Design patterns for ML codebases | ABC, Protocol |

### Training & Models

Skills for building and training deep learning models.

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [PyTorch Lightning](pytorch-lightning.md) | Training loops, modules, and callbacks | Lightning 2.x |
| [Hydra Config](hydra-config.md) | Hierarchical configuration management | Hydra, OmegaConf |
| [Weights & Biases](wandb.md) | Experiment tracking and visualization | wandb |
| [MLflow](mlflow.md) | ML lifecycle and model registry | MLflow |
| [TensorBoard](tensorboard.md) | Training visualization and profiling | TensorBoard |

### Computer Vision

Skills specific to computer vision workflows.

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [OpenCV](opencv.md) | Image processing and video handling | opencv-python |
| [Matplotlib](matplotlib.md) | Visualization and plotting for CV results | matplotlib |
| [ONNX](onnx.md) | Model export and optimization | onnx, onnxruntime, onnxslim |
| [TensorRT](tensorrt.md) | GPU-optimized inference engine building | TensorRT, trtexec |
| [Model Evaluation](model-evaluation.md) | Metrics, mAP/IoU, eval sets, failure analysis | torchmetrics, pycocotools |
| [PydanticAI](pydantic-ai.md) | Type-safe LLM/VLM structured outputs, auto-labeling | pydantic-ai |
| [Hugging Face](huggingface.md) | Pretrained models, fine-tuning, PEFT/LoRA | transformers, datasets, peft |

### Cloud & Deployment

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [AWS SageMaker](aws-sagemaker.md) | ML training and deployment on AWS | sagemaker, boto3 |
| [FastAPI](fastapi.md) | ML model serving APIs | FastAPI, uvicorn |
| [Kubernetes](kubernetes.md) | ML service deployment and orchestration on K8s | kubectl, helm |
| [Gradio](gradio.md) | Interactive ML model demos and prototypes | Gradio |

### Infrastructure & DevOps

Skills for packaging, deploying, and maintaining projects.

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [Pixi](pixi.md) | Package and environment management | Pixi |
| [Docker CV](docker-cv.md) | Containerization for CV workloads | Docker |
| [PyPI](pypi.md) | Python package publishing | build, twine |
| [GCP](gcp.md) | Google Cloud Platform services for ML workflows | gcloud, google-cloud-storage, google-cloud-aiplatform |
| [Vertex AI](vertex-ai.md) | Custom-container training and pipelines on Vertex AI | google-cloud-aiplatform |
| [GitHub Actions](github-actions.md) | CI/CD pipeline configuration | GitHub Actions |
| [GitHub Repo Setup](github-repo-setup.md) | Repository initialization and configuration | gh CLI |
| [Pre-commit](pre-commit.md) | Git hook automation | pre-commit |
| [VS Code](vscode.md) | Editor configuration for ML development | VS Code |
| [DVC](dvc.md) | Data and model version control | DVC |

### Process & Review

Skills for code review and architectural decisions.

| Skill | Description | Key Libraries |
|-------|-------------|---------------|
| [Testing](testing.md) | Test strategy for ML codebases | pytest |
| [Library Review](library-review.md) | Evaluating and selecting ML libraries | N/A |

## Choosing Skills

### By Project Phase

| Phase | Recommended Skills |
|-------|-------------------|
| **Starting a new project** | Master Skill, Code Quality, Pixi, Pydantic |
| **Building models** | PyTorch Lightning, Hydra Config |
| **Training & experiments** | W&B or MLflow, TensorBoard |
| **Preparing for production** | Docker CV, ONNX, Testing, Pre-commit |
| **Publishing & sharing** | PyPI, GitHub Actions, DVC |

### By Role

| Role | Recommended Skills |
|------|-------------------|
| **Researcher** | PyTorch Lightning, Hydra Config, Matplotlib, TensorBoard |
| **ML Engineer** | All Core + Training + Infrastructure |
| **CV Engineer** | OpenCV, PyTorch Lightning, ONNX, Docker CV |
| **DevOps/MLOps** | Docker CV, GitHub Actions, DVC, Pixi |

## Combining Skills

Skills are designed to compose. When you load multiple skills, Claude Code merges their conventions. A few guidelines:

1. **Always include the Master Skill** -- It sets the baseline conventions that other skills build on.
2. **Pick one experiment tracker** -- Use W&B, MLflow, or TensorBoard, not all three (unless migrating).
3. **Infrastructure skills are additive** -- Docker, CI/CD, and Pre-commit never conflict with each other.
4. **Start small, add incrementally** -- Begin with 2-3 skills, then layer on more as the project grows.

## Skill Completeness

Each skill is validated by automated tests that check:

- `SKILL.md` exists and contains 500+ characters of content
- `SKILL.md` includes code examples (fenced code blocks)
- `SKILL.md` includes structural headers
- `README.md` exists and explains the skill's purpose

Run the validation suite:

```bash
uv run pytest tests/test_skills_completeness.py -v
```

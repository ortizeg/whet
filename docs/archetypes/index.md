# Archetypes Overview

Archetypes are project templates for common AI/CV workflows. Each archetype provides a complete directory structure, configuration files, and CI/CD workflows tailored to a specific use case.

## Available Archetypes

| Archetype | Use Case | Key Technologies |
|-----------|----------|-----------------|
| [PyTorch Training](pytorch-training-project.md) | Model training pipelines | Lightning, Hydra, W&B |
| [CV Inference Service](cv-inference-service.md) | Production model serving | FastAPI, ONNX Runtime, Docker |
| [Research Notebook](research-notebook.md) | Jupyter experimentation | Jupyter, matplotlib, Lightning |
| [Library Package](library-package.md) | Reusable Python packages | PyPI, docs, semver |
| [Data Pipeline](data-processing-pipeline.md) | Dataset ETL workflows | Pydantic, parallel processing |
| [Model Zoo](model-zoo.md) | Pretrained model collections | Model cards, benchmarks |

## Common Base Structure

All archetypes share this foundation:

```
{{project_slug}}/
├── .github/workflows/
│   ├── code-review.yml        # Code Review Agent (blocking)
│   └── test.yml               # Test Engineer Agent (blocking)
├── .gitignore
├── .pre-commit-config.yaml
├── pixi.toml                  # Environment + tasks
├── pyproject.toml             # Ruff, mypy, pytest config
├── README.md
├── src/{{package_name}}/
│   └── __init__.py
└── tests/
    └── conftest.py
```

## Choosing an Archetype

- **Training a model?** Use `pytorch-training-project`
- **Deploying a model?** Use `cv-inference-service`
- **Running experiments?** Use `research-notebook`
- **Building a reusable library?** Use `library-package`
- **Processing datasets?** Use `data-processing-pipeline`
- **Managing multiple models?** Use `model-zoo`

## Creating a Project

Use the [Master Skill](../skills/master-skill.md) to initialize a project from an archetype:

```bash
claude

You: "Create a new project using the pytorch-training-project archetype"
```

See the [Creating Projects](../guides/creating-projects.md) guide for a full walkthrough.

# AI/CV Claude Skills

**Production-ready skills framework for computer vision and deep learning projects, designed for Claude Code.**

---

## What Is This?

AI/CV Claude Skills is a curated collection of **32 specialized skills**, **6 agent personas**, and **6 project archetypes** that teach Claude Code how to write production-grade computer vision and deep learning code. Instead of getting generic AI-generated code, you get output that follows the exact patterns, libraries, and conventions used by experienced CV/ML engineers.

Each skill is a focused knowledge module that Claude Code loads on demand. When you tell Claude to "use the PyTorch Lightning skill," it gains deep understanding of Lightning best practices, project structure, and common patterns -- producing code that looks like it was written by a specialist.

## Key Features

- **32 Domain-Specific Skills** -- From PyTorch Lightning training loops to ONNX model export and model evaluation, each skill encodes expert knowledge about a specific tool or pattern.
- **6 Agent Personas** -- Pre-configured behavioral profiles (Expert Coder, ML Engineer, DevOps/Infra, Data Engineer, Code Reviewer, Test Engineer) that adjust Claude's strictness, focus, and output style.
- **6 Project Archetypes** -- Complete project templates for common CV/ML project types: training pipelines, inference services, research notebooks, library packages, data pipelines, and model zoos.
- **Composable by Design** -- Skills combine naturally. A PyTorch training project might use the PyTorch Lightning, Hydra Config, Weights & Biases, and Docker CV skills together.
- **Production Standards** -- Every skill enforces type hints, comprehensive error handling, proper logging, and test coverage. No prototype-quality code.

## Philosophy

This framework is built on three principles:

1. **Specificity over generality.** A skill for PyTorch Lightning does not try to also cover TensorFlow. Narrow scope produces better output.
2. **Convention over configuration.** Each skill prescribes a specific project structure, naming convention, and coding style. Consistency reduces cognitive load.
3. **Composition over inheritance.** Skills are independent modules that you combine for your use case, rather than a monolithic template you trim down.

## Quick Example

```bash
# In your Claude Code session, load skills for a training project
claude "Use the pytorch-lightning, hydra-config, and wandb skills to create
       a ResNet-50 fine-tuning pipeline for a custom image classification dataset"
```

Claude Code will generate:

- A `LightningModule` with proper training/validation steps and metric logging
- Hydra configuration files with structured configs and experiment overrides
- W&B integration with artifact tracking and sweep configurations
- Proper project layout following the PyTorch Training archetype

## Project Structure

```
whet/
    skills/                  # 32 domain-specific skill modules
        master-skill/        # Core conventions for all CV/ML code
        pytorch-lightning/   # PyTorch Lightning training patterns
        pydantic/     # Strict data validation with Pydantic
        ...
    agents/                  # 6 agent persona definitions
        expert-coder/        # High-strictness coding agent
        ml-engineer/         # ML-focused development agent
        ...
    archetypes/              # 6 project templates
        pytorch-training-project/
        cv-inference-service/
        ...
    docs/                    # This documentation
    tests/                   # Framework validation tests
```

## Who Is This For?

- **ML/CV engineers** who use Claude Code and want it to produce code matching their team's standards
- **Researchers** transitioning from notebooks to production code who need guardrails
- **Teams** wanting consistent AI-assisted code generation across projects
- **Solo developers** building CV/ML projects who want expert-level scaffolding

## Getting Started

Ready to dive in? Head to the [Installation](getting-started/installation.md) guide, or jump straight to the [Quick Start](getting-started/quick-start.md) if you already have Claude Code set up.

| Section | Description |
|---------|-------------|
| [Getting Started](getting-started/installation.md) | Install prerequisites, clone the repo, and configure Claude Code |
| [Skills Overview](skills/index.md) | Browse all 32 skills with categories and descriptions |
| [Agents](agents/index.md) | Learn about the 6 agent personas and when to use each |
| [Archetypes](archetypes/index.md) | Explore the 6 project templates |
| [Guides](guides/creating-projects.md) | Step-by-step guides for common workflows |
| [Examples](examples/full-workflow.md) | End-to-end examples with real code |

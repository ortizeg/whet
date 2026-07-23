# Quick Start

This guide gets you from zero to a working project in under five minutes.

## Step 1: Choose Your Skills

Browse the [Skills Overview](../skills/index.md) to find skills relevant to your project. For this quick start, we will build a simple image classification training pipeline using three skills:

- **PyTorch Lightning** -- Training loop and model structure
- **Hydra Config** -- Configuration management
- **Code Quality** -- Linting, formatting, type checking

## Step 2: Install Skills

Install the skills you need into your project:

```bash
mkdir my-classifier && cd my-classifier

# Install skills with whet
whet add pytorch-lightning hydra-config code-quality
```

Start Claude Code — the skills are automatically discovered:

```bash
claude "Create a ResNet image classifier training pipeline.
       Include data loading, augmentations, training configuration,
       and evaluation."
```

## Step 3: Review the Generated Structure

Claude Code will generate a project with this layout:

```
my-classifier/
    configs/
        config.yaml          # Main Hydra config
        model/
            resnet.yaml       # Model-specific config
        data/
            imagenet.yaml     # Dataset config
        trainer/
            default.yaml      # PyTorch Lightning trainer config
    src/
        my_classifier/
            __init__.py
            model.py          # LightningModule with ResNet
            data.py           # LightningDataModule for image data
            transforms.py     # Albumentations/torchvision transforms
            train.py          # Hydra entry point
    tests/
        test_model.py
        test_data.py
    pyproject.toml
    pyproject.toml
```

## Step 4: Configure and Train

Edit the configuration for your dataset:

```yaml
# configs/data/custom.yaml
data:
  root_dir: /path/to/your/dataset
  num_classes: 10
  batch_size: 32
  num_workers: 8
  train_split: 0.8
```

Run training:

```bash
uv run python src/my_classifier/train.py data=custom trainer.max_epochs=50
```

## Step 5: Add More Skills

As your project grows, layer on additional skills:

```bash
# Add experiment tracking
whet add wandb

# Add Docker support
whet add docker-cv

# Add model serving
whet add fastapi

# Add CI/CD
whet add github-actions
```

Each skill is automatically discovered by Claude Code and used to guide code generation.

## Common Skill Combinations

Here are proven combinations for common project types:

### Training Pipeline
```
pytorch-lightning + hydra-config + wandb + code-quality + testing
```

### Inference Service
```
fastapi + onnx + docker-cv + pydantic + code-quality + testing
```

### Research Project
```
pytorch-lightning + matplotlib + hydra-config + tensorboard
```

### Library Package
```
pydantic + code-quality + testing + pypi + pre-commit + github-actions
```

## What Happens Under the Hood

When Claude Code loads a skill, it receives a detailed `SKILL.md` document that contains:

1. **Coding conventions** -- Exact import styles, naming patterns, and project structure
2. **Code templates** -- Annotated examples of correct implementations
3. **Anti-patterns** -- Common mistakes to avoid
4. **Integration patterns** -- How to combine the skill with other tools

Claude Code uses this context to generate code that follows the same patterns consistently across your entire project.

## Next Steps

- [First Project](first-project.md) -- A detailed walkthrough building a complete project
- [Skills Overview](../skills/index.md) -- Browse all available skills
- [Agents](../agents/index.md) -- Learn about agent personas for different task types

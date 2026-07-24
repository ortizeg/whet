# whet

```
  ██╗    ██╗██╗  ██╗███████╗████████╗
  ██║    ██║██║  ██║██╔════╝╚══██╔══╝
  ██║ █╗ ██║███████║█████╗     ██║
  ██║███╗██║██╔══██║██╔══╝     ██║
  ╚███╔███╔╝██║  ██║███████╗   ██║
   ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝   ╚═╝
  sharpen your AI coder
```

[![Tests](https://github.com/ortizeg/whet/actions/workflows/test.yml/badge.svg)](https://github.com/ortizeg/whet/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Install expert skills into AI coding agents.** Works with Claude Code, Google Antigravity, Cursor, and GitHub Copilot.

## Quick Start

```bash
# One-shot (no install needed)
uvx whet list

# Or install permanently
uv tool install whet

# Install all skills to Claude Code
whet install --global

# Add specific skills to your project
whet add pytorch-lightning wandb hydra-config

# Scaffold a new project (also installs the archetype's required skills)
whet init pytorch-training-project
whet init pytorch-training-project --with-recommended   # + recommended skills
whet init pytorch-training-project --no-skills          # files only
```

## What is whet?

whet is a CLI tool that installs curated expert skill definitions into AI coding agents. Each skill is a focused knowledge module (300-600 lines of expert guidance) that teaches your AI coder how to use a specific tool or pattern correctly.

### Flagship Collection: CV/ML

- **32 Skills** — PyTorch Lightning, Pydantic, Docker, ONNX, TensorRT, OpenCV, FastAPI, Hugging Face, AWS SageMaker, Gradio, Kubernetes, Model Evaluation, and more
- **6+ Archetypes** — Training pipelines, inference services, notebooks, packages

### Multi-Platform

| Platform | Status | Skill Format |
|----------|--------|-------------|
| **Claude Code** | Native | SKILL.md (direct copy) |
| **Google Antigravity** | Native | SKILL.md (direct copy) |
| **Cursor** | Adapted | .md rules (converted) |
| **GitHub Copilot** | Adapted | Single aggregated file |

## Commands

```
whet                              # Show banner + help
whet add <skill> [<skill>...]     # Add skill(s) to current project
whet remove <skill>               # Remove a skill
whet install [--global|--local]   # Install full skill collection
whet list [--installed] [--cat]   # Browse available skills
whet search <query>               # Search by name/tag/description
whet info <skill>                 # Show skill details + deps
whet target <platform>            # Set default platform
whet doctor                       # Health check
whet config [key] [value]         # Get/set configuration
```

## How It Works

Skills follow the universal SKILL.md standard — YAML frontmatter for discovery, markdown content for the LLM:

```yaml
---
name: pytorch-lightning
description: >
  PyTorch Lightning training pipeline patterns.
---

# PyTorch Lightning Skill
(300-600 lines of expert guidance...)
```

When you run `whet add pytorch-lightning`, whet copies the skill to your platform's skill directory (e.g., `.claude/skills/`). The AI agent automatically discovers and uses the skill based on your project context.

## Development

```bash
# Clone and install
git clone https://github.com/ortizeg/whet.git
cd whet
uv sync --all-extras

# Run checks
just test
just lint
just typecheck
```

## Contributing

Contributions welcome! See [CLAUDE.md](CLAUDE.md) for how to add new skills or archetypes.

## License

MIT License — see [LICENSE](LICENSE) file.

## Author

Enrique G. Ortiz ([@ortizeg](https://github.com/ortizeg))

# Master Skill

The Master Skill is the entry point and orchestrator for initializing new computer vision and machine learning projects. It acts as a meta-skill that coordinates the selection and composition of all other skills in the repository. Rather than generating code itself, it guides the user through a structured initialization flow -- collecting project metadata, selecting an archetype (such as classification, detection, or segmentation), choosing optional skills, and then delegating to each selected skill to scaffold the full project structure.

The Master Skill reads user inputs including project name, author information, archetype choice, and optional skill selections. Based on these inputs, it determines which skills to activate and in what order, ensuring that dependencies between skills are resolved correctly. For example, selecting the PyTorch Lightning skill automatically activates the Pydantic skill since Lightning modules require Pydantic-based configurations.

## When to Use

- When starting a brand new CV/ML project from scratch.
- When you need a consistent, opinionated project scaffold that follows team conventions.
- When onboarding a new team member who needs a working project skeleton quickly.
- When evaluating which combination of skills best fits a particular project's requirements.

## Key Features

- **Interactive project initialization** -- collects project name, author, description, and license through a guided flow.
- **Archetype selection** -- offers predefined project templates (training, inference, research, library, pipeline, model-zoo) that pre-select relevant skills.
- **Optional skill composition** -- allows toggling additional skills (Docker, CI/CD, DVC, MLflow) beyond the archetype defaults.
- **Dependency resolution** -- ensures that skill prerequisites are satisfied before scaffolding begins.
- **Idempotent generation** -- can be re-run safely to add skills to an existing project without overwriting customized files.
- **Dry-run mode** -- previews the full file tree that would be generated before writing anything to disk.

## Related Skills

- **[Pydantic](../pydantic/)** -- enforces configuration patterns that the Master Skill uses for project metadata and skill parameters.
- **[Pixi](../pixi/)** -- provides the environment and task runner configuration generated during project scaffolding.
- **[Code Quality](../code-quality/)** -- sets up linting, formatting, and type-checking rules as part of the initial project structure.
- **[GitHub Actions](../github-actions/)** -- generates CI/CD workflows when selected as an optional skill during initialization.

# Library Review Skill

The Library Review Skill provides a structured framework for evaluating third-party libraries before they are adopted into a project. In the ML/CV ecosystem, the temptation to `pip install` the latest library is constant -- new training frameworks, augmentation tools, experiment trackers, and serving solutions appear weekly. Without a disciplined evaluation process, projects accumulate poorly maintained dependencies, incompatible licenses, untyped APIs that break mypy, and tightly coupled integrations that resist removal. This skill defines the evaluation criteria, decision process, wrapping strategy, and documentation requirements that every new dependency must pass before entering the codebase.

The evaluation framework scores libraries across multiple dimensions: maintenance health (commit frequency, issue response time, release cadence), typing support (inline types, stub packages, mypy compatibility), license compatibility (permissive vs. copyleft, commercial use restrictions), API quality (consistency, documentation, error handling), and community adoption (GitHub stars, dependent packages, Stack Overflow presence). No single dimension is a dealbreaker, but the combined score must meet a threshold. Libraries that pass evaluation are then categorized by wrapping strategy: direct use for stable, ubiquitous utilities; thin wrappers for well-typed libraries with stable APIs; and full abstraction for volatile or vendor-specific dependencies.

## When to Use

- Before adding any new third-party dependency to the project.
- When a current dependency is deprecated or unmaintained and alternatives need evaluation.
- When comparing two or more libraries that serve the same purpose (e.g., albumentations vs. kornia for augmentations).
- During periodic dependency audits to reassess existing libraries against current criteria.

## Key Features

- **Evaluation criteria matrix** -- scored dimensions for maintenance, typing, license, API quality, and community with weighted thresholds.
- **Decision template** -- structured document capturing the evaluation rationale, alternatives considered, and final recommendation.
- **Wrapping strategy classification** -- three tiers (direct use, thin wrapper, full abstraction) with clear criteria for each.
- **License compatibility guide** -- reference table mapping common OSS licenses to project compatibility, including commercial use considerations.
- **Typing assessment** -- specific checks for py.typed markers, stub packages, mypy plugin support, and generic type coverage.
- **Deprecation playbook** -- process for replacing a library when evaluation reveals a current dependency no longer meets standards.

## Related Skills

- **[Abstraction Patterns](../abstraction-patterns/)** -- implements the wrapping strategy determined by the library review process.
- **[Pydantic](../pydantic/)** -- typing compatibility with Pydantic models is a key evaluation criterion for new libraries.
- **[Code Quality](../code-quality/)** -- mypy compatibility and type stub availability factor heavily into the evaluation score.
- **[Pixi](../pixi/)** -- accepted libraries are added to `pixi.toml` with pinned versions and documented in the lockfile.

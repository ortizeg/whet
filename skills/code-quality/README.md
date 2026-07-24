# Code Quality Skill

The Code Quality Skill enforces consistent code standards across the entire project through two primary tools: Ruff for formatting, linting, and import sorting, and mypy for strict static type checking. Rather than allowing ad-hoc tool configurations to drift across projects, this skill defines an opinionated baseline that balances strictness with practicality -- catching real bugs and enforcing readability without generating noise from irrelevant rules. Every project initialized with this skill shares the same formatting style, the same linting rules, and the same type-checking strictness.

Ruff handles three concerns in a single tool: Black-compatible formatting, isort-compatible import sorting, and linting with a curated rule selection drawn from pyflakes, pycodestyle, bugbear, and other rule sets. The skill specifies which rules to enable, which to ignore (with documented rationale), and how to configure per-file overrides for test files, scripts, and notebooks. Mypy is configured in strict mode with explicit type aliases, plugin support for Pydantic, and per-module overrides where third-party libraries lack stubs.

## When to Use

- At project initialization to establish formatting and linting baselines from day one.
- When adding pre-commit hooks to enforce quality checks before every commit.
- When configuring CI pipelines that gate merges on code quality.
- When onboarding contributors who need clear, automated style guidance.

## Key Features

- **Ruff formatting** -- Black-compatible style with a defined line length, quote style, and trailing comma policy.
- **Ruff linting** -- curated rule sets (F, E, W, I, B, UP, SIM, RUF) with per-rule justifications and explicit ignores.
- **Ruff isort** -- import sorting with known-first-party and known-third-party sections configured for the project.
- **mypy strict mode** -- disallow untyped defs, disallow any generics, warn on unreachable code, and Pydantic plugin enabled.
- **Per-file overrides** -- relaxed rules for test files (assert usage, fixture typing) and scripts (print statements, top-level code).
- **CI integration** -- ready-made commands for running Ruff check, Ruff format, and mypy in GitHub Actions or local task runners.

## Related Skills

- **[Pydantic](../pydantic/)** -- mypy's Pydantic plugin validates model definitions statically, complementing runtime checks.
- **[Pixi](../pixi/)** -- defines task commands (`pixi run lint`, `pixi run typecheck`) that invoke Ruff and mypy with the correct flags.
- **[GitHub Actions](../github-actions/)** -- runs code quality checks as a required CI step before merge.
- **[VS Code](../vscode/)** -- configures editor extensions for Ruff and mypy to surface issues inline during development.

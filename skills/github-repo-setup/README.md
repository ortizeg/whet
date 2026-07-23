# GitHub Repo Setup

## Purpose

Standardizes GitHub repository initialization and configuration for CV/ML projects. Provides reproducible setup for branch protection, PR templates, issue templates, CODEOWNERS, merge strategy, and repository settings using the `gh` CLI.

## When to Use

- Creating a new GitHub repository for a CV/ML project
- Adding branch protection rules to an existing repository
- Setting up PR and issue templates for team collaboration
- Configuring merge strategy and repository settings
- Automating repository configuration across multiple projects

## Key Features

- Repository initialization with standard structure
- Branch protection via `gh api` commands
- PR template with test plan and checklist
- Issue templates (bug report, feature request)
- CODEOWNERS for automated review routing
- Squash merge as default strategy
- Full setup automation script
- Pydantic configuration model for repo settings
- Follow gitflow workflow

## Related Skills

- **GitHub Actions** — CI/CD workflows that become required status checks
- **Pre-commit** — Local git hooks complementing CI-level checks
- **Code Quality** — Linting and type checking enforced by branch protection

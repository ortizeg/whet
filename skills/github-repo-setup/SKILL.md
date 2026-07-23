---
name: github-repo-setup
description: >
  Use this skill when initializing or configuring a GitHub repository for a CV/ML project
  — creating the repo, branch protection rules, PR and issue templates, CODEOWNERS, merge
  strategy, and gh CLI automation. Reach for it any time you'd otherwise click through
  GitHub settings by hand or ask "how should I set up this repo", even if the user just
  says "create the repo" or "protect main". For the CI/CD workflows that run on those
  branches see github-actions.
---

# GitHub Repository Setup Skill

Best practices for initializing and configuring GitHub repositories for CV/ML projects. Covers repository creation, branch protection, PR templates, issue templates, CODEOWNERS, merge strategy, and automation via the `gh` CLI.

## Repository Initialization

Create repositories with consistent structure from the start.

```bash
# Create a new public repo with README, LICENSE, and .gitignore
gh repo create my-cv-project \
  --public \
  --clone \
  --license MIT \
  --gitignore Python \
  --description "Computer vision training pipeline"

cd my-cv-project

# Initialize with standard structure
mkdir -p src/my_cv_project tests docs .github/workflows .github/ISSUE_TEMPLATE
touch src/my_cv_project/__init__.py
touch src/my_cv_project/py.typed
touch tests/__init__.py
```

### Standard Files Checklist

Every repository must have:

- `README.md` — Project overview, quick start, badges
- `LICENSE` — MIT for open-source CV/ML projects
- `.gitignore` — Python template + ML-specific exclusions (model weights, datasets, wandb/)
- `pyproject.toml` — Project metadata and tool configuration
- `pixi.toml` — Environment and dependency management

## Branch Protection

Protect `main` to enforce code quality through pull requests.

```bash
# Enable branch protection on main
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test", "lint", "docs"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_linear_history": false
}
EOF
```

### Key Settings Explained

| Setting | Value | Rationale |
|---------|-------|-----------|
| `required_approving_review_count` | 1 | Minimum review gate without blocking solo developers |
| `strict` status checks | true | Branch must be up-to-date before merging |
| `enforce_admins` | true | Admins follow the same rules |
| `dismiss_stale_reviews` | true | New pushes invalidate old approvals |
| `allow_force_pushes` | false | Protect commit history |

## Pull Request Template

Standardize PR descriptions to capture context for reviewers.

```markdown
<!-- .github/PULL_REQUEST_TEMPLATE.md -->

## Summary
<!-- Brief description of changes -->

## Changes
-

## Test Plan
- [ ] `pixi run test` passes
- [ ] `pixi run lint` passes
- [ ] `pixi run typecheck` passes
- [ ] `pixi run docs-build` succeeds

## Checklist
- [ ] Tests added/updated for new functionality
- [ ] Documentation updated if needed
- [ ] No new type errors introduced
```

## Issue Templates

Provide structured templates for bug reports and feature requests.

### Bug Report Template

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug or unexpected behavior
labels: ["bug"]
body:
  - type: textarea
    id: description
    attributes:
      label: Description
      description: Clear description of the bug
    validations:
      required: true
  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: Minimal steps to reproduce the issue
      value: |
        1. ...
        2. ...
        3. ...
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What you expected to happen
    validations:
      required: true
  - type: input
    id: version
    attributes:
      label: Version
      description: Package version or commit hash
    validations:
      required: true
  - type: dropdown
    id: os
    attributes:
      label: Operating System
      options:
        - Linux
        - macOS
        - Windows
    validations:
      required: true
```

### Feature Request Template

```yaml
# .github/ISSUE_TEMPLATE/feature_request.yml
name: Feature Request
description: Suggest a new feature or improvement
labels: ["enhancement"]
body:
  - type: textarea
    id: problem
    attributes:
      label: Problem Statement
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: solution
    attributes:
      label: Proposed Solution
      description: How should this work?
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives Considered
      description: Other approaches you've considered
    validations:
      required: false
```

## CODEOWNERS

Define ownership for code review routing.

```
# .github/CODEOWNERS
# Default owner for everything
* @org/ml-team

# Specific ownership
/skills/         @org/ml-team
/agents/         @org/ml-team
/archetypes/     @org/ml-team
/docs/           @org/docs-team
/.github/        @org/devops-team
pixi.toml        @org/devops-team
pyproject.toml   @org/ml-team
```

### CODEOWNERS Syntax

| Pattern | Meaning |
|---------|---------|
| `*` | Everything (default) |
| `/docs/` | Only the top-level `docs/` directory |
| `*.py` | All Python files anywhere |
| `/src/models/` | Specific subdirectory |
| `@user` | Individual GitHub user |
| `@org/team` | GitHub team |

## Merge Strategy

Use squash merge as the default for clean history.

```bash
# Configure repository to use squash merge by default
gh api repos/{owner}/{repo} \
  --method PATCH \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=false \
  --field squash_merge_commit_title=PR_TITLE \
  --field squash_merge_commit_message=PR_BODY \
  --field delete_branch_on_merge=true
```

### When to Use Each Strategy

| Strategy | When to Use |
|----------|-------------|
| **Squash merge** (default) | Feature branches, bug fixes, most PRs |
| **Merge commit** | Release branches, long-lived branches with meaningful history |
| **Rebase merge** | Never for this framework (loses PR association) |

## Required Status Checks

Map CI workflows to branch protection status checks.

```bash
# The "contexts" in branch protection must match workflow job names
# In .github/workflows/test.yml, the job name "test" becomes the status check

# Example: workflows/test.yml
# jobs:
#   test:        <-- This becomes the status check name "test"
#     runs-on: ubuntu-latest
#     steps: ...

# Example: workflows/lint.yml
# jobs:
#   lint:        <-- This becomes "lint"
#     runs-on: ubuntu-latest
#     steps: ...
```

To find available status check names after a first push:

```bash
# List recent check runs
gh api repos/{owner}/{repo}/commits/main/check-runs \
  --jq '.check_runs[].name'
```

## GitHub Pages Setup

Enable GitHub Pages deployment from GitHub Actions.

```bash
# Enable Pages with Actions as the build source
gh api repos/{owner}/{repo}/pages \
  --method POST \
  --field build_type=workflow

# Verify Pages is enabled
gh api repos/{owner}/{repo}/pages --jq '.html_url'
```

## Repository Settings

Configure repository-level settings for consistency.

```bash
# Disable unused features, enable useful defaults
gh api repos/{owner}/{repo} \
  --method PATCH \
  --field has_wiki=false \
  --field has_projects=false \
  --field has_discussions=false \
  --field delete_branch_on_merge=true \
  --field allow_auto_merge=true
```

## Full Setup Script

Automate the complete repository configuration in one shot.

```bash
#!/usr/bin/env bash
# setup-repo.sh — Configure a GitHub repository with best practices
set -euo pipefail

OWNER="${1:?Usage: setup-repo.sh OWNER REPO}"
REPO="${2:?Usage: setup-repo.sh OWNER REPO}"

echo "Configuring $OWNER/$REPO..."

# 1. Repository settings
gh api "repos/$OWNER/$REPO" \
  --method PATCH \
  --field has_wiki=false \
  --field has_projects=false \
  --field delete_branch_on_merge=true \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=false \
  --field squash_merge_commit_title=PR_TITLE \
  --field squash_merge_commit_message=PR_BODY

# 2. Branch protection
gh api "repos/$OWNER/$REPO/branches/main/protection" \
  --method PUT \
  --input - <<'PROTECTION'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test", "lint"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
PROTECTION

echo "Done. Repository configured."
```

## Pydantic Configuration Model

Define repository configuration as a typed, validated model.

```python
from __future__ import annotations

from pydantic import BaseModel, Field


class BranchProtection(BaseModel):
    """Branch protection rule configuration."""

    required_approvals: int = Field(default=1, ge=0, le=6)
    dismiss_stale_reviews: bool = True
    require_code_owner_reviews: bool = True
    enforce_admins: bool = True
    allow_force_pushes: bool = False
    required_status_checks: list[str] = Field(default_factory=lambda: ["test", "lint"])


class MergeStrategy(BaseModel):
    """Repository merge strategy configuration."""

    allow_squash: bool = True
    allow_merge_commit: bool = False
    allow_rebase: bool = False
    delete_branch_on_merge: bool = True
    squash_title: str = "PR_TITLE"
    squash_message: str = "PR_BODY"


class RepoConfig(BaseModel):
    """Complete repository configuration."""

    owner: str
    name: str
    description: str = ""
    visibility: str = Field(default="private", pattern=r"^(public|private)$")
    license: str = "MIT"
    branch_protection: BranchProtection = Field(default_factory=BranchProtection)
    merge_strategy: MergeStrategy = Field(default_factory=MergeStrategy)
    has_wiki: bool = False
    has_projects: bool = False
    has_discussions: bool = False
    enable_pages: bool = False
    codeowners: dict[str, list[str]] = Field(
        default_factory=lambda: {"*": ["@owner"]},
        description="Pattern to owners mapping",
    )
```

## Best Practices

1. **Protect main from day one** — Enable branch protection before the first collaborator joins.
2. **Use squash merge** — Keeps main history clean with one commit per PR.
3. **Delete branches on merge** — Prevents stale branch accumulation.
4. **Require status checks** — Never merge without CI passing.
5. **Use CODEOWNERS** — Automate review assignments and enforce ownership.
6. **Template everything** — PR templates, issue templates, and CONTRIBUTING.md reduce friction.
7. **Disable unused features** — Wiki, Projects, Discussions add noise if unused.
8. **Enable auto-merge** — Let PRs merge automatically when all checks pass.
9. **Script your setup** — Use `gh` CLI scripts so configuration is reproducible across repos.
10. **Review protection quarterly** — As the team grows, adjust required reviewers and status checks.

## Anti-Patterns

- **No branch protection** — Pushing directly to main risks breaking the project.
- **Too many required reviewers** — More than 2 for small teams creates bottlenecks.
- **Merge commits for feature branches** — Pollutes history with merge noise.
- **Manual repository setup** — Click-through configuration is not reproducible.
- **Skipping PR templates** — Leads to empty PR descriptions and lost context.
- **Overly broad CODEOWNERS** — `* @everyone` means no one owns anything.
- **Ignoring stale reviews** — Approving code that has changed since review is dangerous.
- **Force pushing to main** — Rewrites shared history and breaks collaborator checkouts.

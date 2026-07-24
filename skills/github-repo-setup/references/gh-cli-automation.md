# gh CLI Automation

Scope: scripting repository configuration end to end — repository settings, GitHub Pages,
a one-shot setup script, and a typed Pydantic model of the configuration.

## Contents

- [Repository Settings](#repository-settings)
- [GitHub Pages Setup](#github-pages-setup)
- [Full Setup Script](#full-setup-script)
- [Pydantic Configuration Model](#pydantic-configuration-model)

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

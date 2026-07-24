# Branch Protection and Merge Strategy

Scope: protecting `main` via the GitHub API, mapping CI job names to required status
checks, and configuring the repository's merge strategy.

## Contents

- [Enabling Branch Protection](#enabling-branch-protection)
- [Key Settings Explained](#key-settings-explained)
- [Merge Strategy](#merge-strategy)
- [When to Use Each Strategy](#when-to-use-each-strategy)
- [Required Status Checks](#required-status-checks)

## Enabling Branch Protection

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

## Key Settings Explained

| Setting | Value | Rationale |
|---------|-------|-----------|
| `required_approving_review_count` | 1 | Minimum review gate without blocking solo developers |
| `strict` status checks | true | Branch must be up-to-date before merging |
| `enforce_admins` | true | Admins follow the same rules |
| `dismiss_stale_reviews` | true | New pushes invalidate old approvals |
| `allow_force_pushes` | false | Protect commit history |

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

## When to Use Each Strategy

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
